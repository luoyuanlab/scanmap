import torch
from torch import nn
from collections import  defaultdict
import numpy as np
import utils
import pandas as pd
from sklearn.metrics import accuracy_score

## this implements ScanMap transductive learning with demographics information and orthogonality constraints
## assumes X is passed as [Xtrain; Xtest], and only y_train is passed
class ScanMap(nn.Module):
    def __init__(self, X, cf = None, cfval = None, y = None, yval=None,
                 k = 10, n_iter = 10, eps = 1e-7,
                 floss = 'l2', weight_decay = 1e-5, wortho=1, wcls=0.1, C=1,
                 lr = 1e-2, verbose = False, seed=None, fn=None,
                 device=torch.device('cpu')):
        super(ScanMap, self).__init__()
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
        self.n_iter = n_iter
        self.k = k
        self.floss = floss
        self.weight_decay = weight_decay
        self.lr = lr
        self.verbose = verbose
        self.eps = eps
        self.wcls = wcls
        self.wortho = wortho
        self.fn = fn
        self.C = C
        self.decomposed = False
        self.dcf = cf.shape[1]
        self.device = device
        self.celoss = -1
        self.report = defaultdict(list)
        self.__initfact__(X, cf, cfval, y, yval)

    def __initfact__(self, X, cf, cfval, y, yval):
        self.n,self.m = X.shape
        self.ntr = len(y)
        self.nval = len(yval)
        self.nte = self.n - self.ntr - self.nval
        self.X = torch.from_numpy(X).float().to(self.device)
        self.cf = torch.from_numpy(cf).float().to(self.device)
        self.cfval = torch.from_numpy(cfval).float().to(self.device)
        
        self.scale = torch.mean(self.X) / self.k
        Wtr = torch.abs(torch.rand([self.ntr,self.k]) * self.scale).to(self.device)
        Wval = torch.abs(torch.rand([self.nval,self.k]) * self.scale).to(self.device)
        Wte = torch.abs(torch.rand([self.nte,self.k]) * self.scale).to(self.device)
        H = torch.abs(torch.rand([self.k,self.m])).to(self.device)
        self.Wtr = torch.nn.Parameter(Wtr)
        self.Wval = torch.nn.Parameter(Wval)
        self.Wte = torch.nn.Parameter(Wte)
        self.H = torch.nn.Parameter(H)
        self.identity = torch.eye(self.k, device=self.device)
        
        if self.floss == 'l2':
            self.loss_fac = utils.l2
        elif self.floss == 'kl':
            self.loss_fac = utils.kl_div

        self.y = torch.from_numpy(y).long().to(self.device)
        w = 1 / pd.Series(y).value_counts(normalize=True).sort_index().to_numpy()
        self.loss_cls = nn.CrossEntropyLoss().to(self.device) # weight=torch.from_numpy(w).float()
        self.fc = nn.Linear(self.k+self.dcf, len(np.unique(y))).to(self.device)
            
        self.yval = torch.from_numpy(yval).long().to(self.device)
        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)

    def to(self,device):
        self.device = device
        self.X = self.X.to(device)
        self.cf = self.cf.to(device)
        self.cfval = self.cfval.to(device)
        self.y = self.y.to(device)
        self.yval = self.yval.to(device)        
        return super(ScanMap, self).to(device)

    def plus(self,X):
        X[X < 0] = 0 # self.eps
        return X

    def __autograd__(self,epoch):
        """
           autograd update, with gradient projection
        """
        self.opt.zero_grad()
        A = torch.cat([self.Wtr, self.Wval, self.Wte])
        l = self.loss_fac(A @ self.H, self.X) * self.k * self.n #   01/24/20
        # add l2 regularization for orthogonality
        AtA = torch.mm(torch.t(A), A)
        if torch.mean(AtA) > self.eps:
            AtA = AtA/torch.mean(AtA)
        l += self.loss_fac(AtA/self.k, self.identity) * self.wortho * self.k # scale invariant orthogonal
        if self.y is not None:
            self.celoss = self.loss_cls(self.fc(torch.cat([self.Wtr, self.cf], 1)), self.y)
            l = l + self.celoss * self.wcls * self.n # 01/24/20
            # print('cross entropy: %.4f' % (self.loss_cls(self.fc(self.Wtr), self.y)))
            for p in self.fc.parameters():
                # has two parameters (cls #, ft #) and (cls #), they are the weight and biases, /self.k to add normalized weight regularization
                l = l + p.pow(2).sum() * self.C 
                # print('complexity: %.4f' % (p.pow(2).sum()))

        l.backward()
        self.opt.step()
        ## grad projection
        self.Wtr.data = self.plus(self.Wtr.data)
        self.Wval.data = self.plus(self.Wval.data)
        self.Wte.data = self.plus(self.Wte.data)
        self.H.data = self.plus(self.H.data)
        return l.item()


    def __update__(self,epoch):
        l = self.__autograd__(epoch)

        self.report['epoch'].append(epoch)
        self.report['loss'].append(l)
        if self.verbose and epoch % 100 == 0:
            print("%d\tloss: %.4f"%(epoch,l))


    def fit(self):
        it = range(self.n_iter)
        # for autograd solver
        best_val_acc = 0
        for e in it:
            self.__update__(e)
            # here using pinverse seems to mess up GPU/CPU, it's really the pinverse that's taking a lot of CPU.
            if e >= self.n_iter - 1:
                y_val_pred = self.__predict__(self.Wval, self.cfval)
                acc = accuracy_score(self.yval.detach().cpu().numpy(),
                                    y_val_pred.detach().cpu().numpy())
                if acc > best_val_acc:
                    best_val_acc = acc
                    if self.fn is not None:
                        torch.save({'epoch': e+1,
                                    'state_dict': self.state_dict(),
                                    'optimizer': self.opt.state_dict(),
                                    'best_val_acc': best_val_acc,
                                    'report': self.report,
                                    'celoss': self.celoss,
                        }, self.fn)
                        # print('best_val_acc: %.4f, test_acc: %.4f' % (best_val_acc, test_acc))
        self.decomposed = True
        return self

    def show_report(self):
        return pd.DataFrame(self.report)

    def fit_transform(self):
        if not self.decomposed:
            self.fit()
            # detach all params including the linear fc layer
            for p in self.parameters():
                p.requires_grad = False
        if self.device.type == 'cuda':
            return [self.Wtr.detach().cpu().numpy(), self.Wval.detach().cpu().numpy(), self.Wte.detach().cpu().numpy(), self.H.detach().cpu().numpy()]
        else:
            return [self.Wtr.detach().numpy(), self.Wval.detach().numpy(), self.Wte.detach().numpy(), self.H.detach().numpy()]

    def predict(self, w, cf):
        w = torch.from_numpy(w).float()
        cf = torch.from_numpy(cf).float()                        
        if self.device.type == 'cuda':
            w = w.to(self.device)
            cf = cf.to(self.device)
        y = self.__predict__(w, cf)
        if self.device.type == 'cuda':
            return y.detach().cpu().numpy()
        else:
            return y.detach().numpy()

    def __predict__(self, w, cf):
        y = torch.argmax(self.fc(torch.cat([w, cf],1)), dim=1)
        return y

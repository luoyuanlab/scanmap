import pandas as pd
import pickle
from scipy.sparse import coo_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, accuracy_score
from math import floor, ceil
import numpy as np
import os
import sys
import importlib
import torch
from ScanMap import ScanMap
import warnings
warnings.filterwarnings(action='ignore')
import torch.nn as nn
import getopt

opts, extraparams = getopt.getopt(sys.argv[1:], 's:c:i:r:', 
                                  ['seed=', 'config=', 'iter=', 'lr='])

print(sys.argv)
devstr = 'cuda'
config = 'please specify your own configuration string describing, e.g., germline mutations, filtering thresholds in pre-processing steps'
niter = 4000
lr=0.01
seed = 1

for o,p in opts:
    if o in ['-s', '--seed']:
        seed = int(p)
    if o in ['-c', '--config']:
        config = p
    if o in ['-i', '--iter']:
        niter = int(p)
    if o in ['-r', '--lr']:
        lr = float(p)

dn = 'please specify your root genetic data directory'
# the pickle file contains the feature matrix and the label
f = open('%s/%s/tcga.pik' % (dn, config), 'rb')
[tcga_mat, y] = pickle.load(f)
f.close()

print('matrix shape: {0} x {1}'.format(*tcga_mat.shape))

# read in demographic information as confounding variables
dnphe = 'please specify your root phenotype data directory'
# the demographic csv should have case_id, gender, race columns
pts = pd.read_csv('%s/pt_demo.csv' % (dnphe))
pts.race = pts.race.str.replace(' ', '_')
pts.set_index('case_id', inplace=True)
pts = pd.get_dummies(pts[['gender', 'race']])
pts.drop(['gender_FEMALE', 'race_Unknown'], axis=1, inplace=True)

# make sure that the genetic data and confounding variables match each other
pts_sel = pts.loc[tcga_mat.index]
pts_sel.fillna(0, inplace=True)
sel_pts = np.array(pts_sel)

# read in train val test split, use pre-generated indices for reproducibility
train_indices = pd.read_csv('%s/train_indices_0.2val_0.2te.csv' % (dn), header=None) #  _5run
test_indices = pd.read_csv('%s/test_indices_0.2val_0.2te.csv' % (dn), header=None)
val_indices = pd.read_csv('%s/val_indices_0.2val_0.2te.csv' % (dn), header=None)
X = np.array(tcga_mat)
y, yuniques = pd.factorize(y, sort=True)



r = 0
ncs = range(50,501,50) 
X = X.astype(float)
device = torch.device(devstr)

train_index = train_indices[r]; val_index = val_indices[r]; test_index = test_indices[r]
X_train, X_val, X_test = X[train_index], X[val_index], X[test_index]
y_train, y_val, y_test = y[train_index], y[val_index], y[test_index]
pts_tr, pts_val, pts_te = sel_pts[train_index], sel_pts[val_index], sel_pts[test_index]    

print('nc,wcls,C,best iter,tr acc,val acc,te acc,w2,b2,celoss,mse,mse tr,mse val,mse te')
for nc in ncs:
    for C in [0.01, 0.1, 1, 10, 100]: # 0.001,  , 1000
        for wcls in [0.1, 0.5, 1, 2, 10]: #  
            fn = '%s/%s/scanmap%d/s%d/scanmap_k%d_wcls%s_C%s.p' % (dn, config, niter, seed, nc, wcls, C)
            m = ScanMap(np.vstack((X_train, X_val, X_test)), cf = pts_tr, cfval = pts_val, y = y_train, yval = y_val, k=nc, n_iter=4000, weight_decay=0, lr=lr, wcls=wcls, C=C, seed=2*722019+seed, fn=fn, device=device) # 2*722019+seed is just to have a large odd number for seeding that is recommended for generating random numbers, fixed for reproducibility
            [X_tr_nmf, X_val_nmf, X_te_nmf, H] = m.fit_transform()

            chkpt = torch.load(fn)
            m.load_state_dict(chkpt['state_dict'])
            best_iter = chkpt['epoch']
            accval = chkpt['best_val_acc']
            m.eval()

            y_tr_pred = m.predict(X_tr_nmf, pts_tr)
            y_te_pred = m.predict(X_te_nmf, pts_te)
            acctr = accuracy_score(y_train, y_tr_pred) 
            accte = accuracy_score(y_test, y_te_pred) 
            
            w2 = np.square(m.state_dict()['fc.weight'].cpu().numpy()).sum(axis=None)
            b2 = np.square(m.state_dict()['fc.bias'].cpu().numpy()).sum(axis=None)
            w = 1 / pd.Series(y_train).value_counts(normalize=True).sort_index().to_numpy()
            vce = chkpt['celoss']

            err = np.vstack((X_train, X_val, X_test)) - np.vstack((X_tr_nmf, X_val_nmf, X_te_nmf)) @ H
            err_tr = X_train - X_tr_nmf @ H
            err_val = X_val - X_val_nmf @ H
            err_te = X_test - X_te_nmf @ H
            mse = np.square(err).mean(axis=None)
            mse_tr = np.square(err_tr).mean(axis=None)
            mse_val = np.square(err_val).mean(axis=None)
            mse_te = np.square(err_te).mean(axis=None)
            
            print('%d,%s,%s,%d,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f' % 
                  (nc, wcls, C, best_iter, acctr, accval, accte, w2, b2, vce, mse, mse_tr, mse_val, mse_te))

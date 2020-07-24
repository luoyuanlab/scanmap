# ScanMap

### Requirements
Code is written in Python (3.7.3) and requires PyTorch (1.0.0).

### Data
In this experiment, we have used the dataset from The Cancer Genome Atlas (TCGA), which can be downloaded at https://portal.gdc.cancer.gov/

### Analysis
To perform ScanMap analysis on germline TCGA data, run
```
CUDA_VISIBLE_DEVICES=0 python high_germline_scanmap.py -c<config string> -i4000 -r0.01 -s1 >result.txt 2>&1 &
```

The code `high_germline_scanmap.py` is a wrapper code that takes in a pickle file consisting of subject-by-gene (or subject-by-pathway) matrix, reads in confounding variables corresponding to the subjects, calls the `ScanMap` class in `ScanMap.py` to perform supervised confounding aware NMF for polygenic risk modeling.

The meanings of the parameters are defined in `high_germline_scanmap.py`. This code by defaults uses visible GPU.

### Citation
```
@inproceedings{luo2020scanmap,
  title={ScanMap: Supervised Confounding Aware Non-negative Matrix Factorization for Polygenic Risk Modeling},
  author={Luo, Yuan and Mao, Chengsheng},
  booktitle={Machine Learning for Healthcare Conference},
  year={2020}
}
```

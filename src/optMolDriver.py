import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler
from rdkit import Chem

# load the data
#YX = torch.load('/Users/eldadhaber/Dropbox/ComputationalBio/Data/rdk/dataset_rdkit.pt')
#Y = YX[:,0]
#X = YX[:,1:]

import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

# load the data
#YX = torch.load('/Users/eldadhaber/Dropbox/ComputationalBio/Data/rdk/dataset_rdkit.pt')
E = torch.load('/Users/eldadhaber/Dropbox/ComputationalBio/Data/rdk/energies.pt')
X = torch.load('/Users/eldadhaber/Dropbox/ComputationalBio/Data/rdk/finger_prints_1024.pt')
s = X.sum(dim=1)

#U,S,V = torch.svd(X)
#X = U*(1/S)



#Y = YX[:, 0]
#X = YX[:, 1:]



def getSimilarDataBin(ind,X,s,R,N):

    x = X[ind,:].t()
    r = (X@x)/s
    d = -torch.abs(r-R)
    vals, indices = torch.topk(d, k=2*N, dim=0)
    ii = torch.randperm(2*N)
    vals = vals[ii[:N]]
    indices = indices[ii[:N]]


    return indices


def getSimilarData(ind,X,s,R=128,N=64):

    x = X[ind,:]
    #r = torch.pow(X - x,2).sum(dim=1)
    r = torch.abs(X - x).sum(dim=1)

    d = -torch.abs(r-R)
    vals, indices = torch.topk(d, k=2*N, dim=0)
    ii = torch.randperm(2*N)
    vals = vals[ii[:N]]
    indices = indices[ii[:N]]

    return indices

def getSimilarDataTanimoto(ind,X,s,R,N):

    x = X[ind,:].t()
    dp = X@x
    nz = torch.sum((X+x)>0,dim=1)
    T  = dp/nz
    d = -torch.abs(T-R)
    vals, indices = torch.topk(d, k=2*N, dim=0)
    ii = torch.randperm(2*N)
    vals = vals[ii[:N]]
    indices = indices[ii[:N]]


    return indices


def gridSearch(ind0, X, s, E,  N=10, niter=20, R=128):

    ind = ind0
    f = E[ind]
    print(0, f.item())
    f = f[0]
    IND = torch.tensor([ind])
    hist = torch.zeros(niter,2)
    for i in range(niter):
        # Sample points
        #inds = getSimilarDataBin(ind, X, s, R, N)
        inds = getSimilarData(ind, X, s, R, N)
        #inds = getSimilarDataTanimoto(ind, X, s, R, N)
        IND = torch.cat((IND,inds),dim=0)
        # Evaluate function
        ftry = E[inds]
        #
        # Find if we improve
        jmin = torch.argmin(ftry)
        #print(jmin, f[jmin])
        if ftry[jmin]<f:
            ind = inds[jmin]
            f   = ftry[jmin]
        else:
            R = R/1.05

        print(i, f.item(), ftry[jmin].item(), R)
        hist[i,0] = f
        hist[i,1] = R
        if R < 8:
            print('Total function evaluations ', N * i)
            return ind, f, hist, IND

    print('Total function evaluations ', N * niter)
    return ind, f, hist, IND


ind0 = torch.randint(X.shape[0],(1,)).item() # The maximal index 19237 #
N = 64
niter = 250
ind, d, hist, IND = gridSearch(ind0, X, s, E, N=N, niter=niter, R=128)

# Compare to random sumpling
ii = torch.randint(49184,(len(IND),))
nbins = 64
x = torch.linspace(E[ii,0].min(),E[ii].max(),nbins)
h = torch.histc(E[ii,0], bins=nbins, min=E[ii,0].min(), max=E[ii,0].max())
plt.plot(x,h,'.-r')
x = torch.linspace(E[IND,0].min(),E[IND,0].max(),nbins)
h = torch.histc(E[IND,0], bins=nbins, min=E[IND,0].min(), max=E[IND,0].max())
plt.plot(x,h,'.-b')


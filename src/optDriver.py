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
YX = torch.load('/Users/eldadhaber/Dropbox/ComputationalBio/Data/rdk/dataset_rdkit.pt')

Y = YX[:,0]
X = YX[:,1:]


def indexData(X, numnei=32):
    ##
    d = torch.pow(X,2).sum(dim=1, keepdim=True)
    D = torch.relu(d + d.t() - 2*X@X.t())
    sigma = torch.std(D)
    D = torch.exp(-D/sigma**2)
    ###
    vals, indices = torch.topk(D, k=numnei, dim=1)
    nd = D.shape[0]
    I = torch.ger(torch.arange(nd), torch.ones(numnei, dtype=torch.long))
    I = I.view(-1)
    J = indices.view(-1).type(torch.LongTensor)
    IJ = torch.stack([I, J], dim=1)
    one = torch.ones(IJ.shape[0])

    A = torch.sparse_coo_tensor(IJ.t(), one, (D.shape[0], D.shape[1]))
    return A


def getNfarNeighbours(ind,A,k,n):

    n = A.shape[0]
    x = torch.zeros(n)
    x[ind] = 1

    Ind = torch.LongTensor([ind])
    for i in range(k-1):
        x = torch.mv(A,x)
    ij = x.nonzero(as_tuple=True)

    x = torch.mv(A,x)
    ijLast = x.nonzero(as_tuple=True)

    combined = torch.cat((ij, ijLast))
    uniques, counts = combined.unique(return_counts=True)
    diff   = uniques[counts == 1]
    #inter = uniques[counts > 1]

    L = len(diff)
    if L<n:
        Xout = X[diff,:]
    else:
        t        = torch.rand(L)
        ts, inds = t.sort()
        indout   = inds[:n]
        Xout     = X[indout]

    return Xout


def getSimilarData(x,X,R,N):

    r = torch.pow(X - x,2).sum(dim=1)
    d = -torch.abs(r-R)
    vals, indices = torch.topk(d, k=N, dim=0)

    xout = X[indices,:]

    return xout


# Try an example
tryCode = False
if tryCode:
    x = torch.zeros(1,2)
    X = torch.randn(1000,2)

    xs = getSimilarData(x,X,0.5,100)

    plt.plot(X[:,0],X[:,1],'.')
    plt.plot(x[:,0],x[:,1],'xb')
    plt.plot(xs[:,0],xs[:,1],'xr')


# Function Evaluation
def energyFunction(x,param=[100,1.0]):

    # The banana function
    f = param[0]*(x[:,1] - x[:,0]**2)**2 + (param[1]-x[:,0])**2

    return f


# Optimization Code
def gridSearch(x0, X , N=10, niter=20, R=4):

    x = x0
    f = energyFunction(x)
    hist = torch.zeros(niter,2)
    XX   = torch.zeros(niter+1,2)
    XX[0,:] = x
    for i in range(niter):
        # Sample points
        xtry = getSimilarData(x, X, R, N)
        # Evaluate function
        ftry = energyFunction(xtry)
        #
        # Find if we improve
        jmin = torch.argmin(ftry)
        #print(jmin, f[jmin])
        if ftry[jmin]<f:
            x = xtry[jmin,:]
            f = ftry[jmin]
        else:
            R = R/2

        print(i, f.item(), ftry[jmin].item(), R)
        hist[i,0] = f
        hist[i,1] = R
        XX[i+1,:] = x

    return x, f, XX

tryCode = True
if tryCode:
    x = torch.ones(1,2)*4
    X = torch.randn(1000,2)

    x, f, XX = gridSearch(x, X, N=16, niter=20, R=4)

    # for plotting
    fX = energyFunction(X)
    fXX = energyFunction(XX)
    plt.scatter(X[:,0],X[:,1], c=torch.log(fX+1e-4), alpha=0.2)
    plt.plot(XX[:,0],XX[:,1],'xr')

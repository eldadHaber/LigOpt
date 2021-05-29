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

Y = YX[:, 0]
X = YX[:, 1:]


def indexData(X, numnei=32):
    ##
    d = torch.pow(X, 2).sum(dim=1, keepdim=True)
    D = torch.relu(d + d.t() - 2 * X @ X.t())
    sigma = torch.std(D)
    D = torch.exp(-D / sigma ** 2)
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


def getNfarNeighbours(ind, A, X, k, nout):
    n = A.shape[0]
    x = torch.zeros(n)
    x[ind] = 1
    ij = x.squeeze().nonzero(as_tuple=True)[0]
    for i in range(k):
        x = torch.mv(A, x)
        ijIter = x.squeeze().nonzero(as_tuple=True)[0]
        ij     = torch.cat((ij, ijIter))

    uniques, counts = ij.unique(return_counts=True)

    c, inds = torch.sort(counts)
    Xout = X[uniques[inds[:nout]],:]

    return Xout, uniques[inds[:nout]]



#A = indexData(X, numnei=32)

def getSimilarData(ind,X,R,N):

    x = X[ind,:]
    r = torch.pow(X - x,2).sum(dim=1)
    d = -torch.abs(r-R)
    vals, indices = torch.topk(d, k=N, dim=0)

    xout = X[indices,:]

    return xout, indices


def gridSearch(ind0, YX, N=10, niter=20, R=64):

    ind = ind0
    yx = YX[ind,:]
    f = yx[0]
    hist = torch.zeros(niter,2)
    for i in range(niter):
        # Sample points
        #_, inds = getNfarNeighbours(ind, A, X, k, N)
        _, inds = getSimilarData(ind, X, R, N)
        # Evaluate function
        YXtry = YX[inds,:]
        ftry = YXtry[:,0]
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
            return ind, f, hist

    return ind, f, hist


ind0 = 19237 #torch.randint(YX.shape[0],(1,)).item()
ind, d, hist = gridSearch(ind0, YX, N=64, niter=250, R=128)
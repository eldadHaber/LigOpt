import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import math
import torch.autograd.profiler as profiler

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

    xs = getSimilarData(x,X,2.5,100)

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

        print(i, f, ftry[jmin], R)
        hist[i,0] = f
        hist[i,1] = R
        XX[i+1,:] = x

    return x, f, XX

tryCode = True
if tryCode:
    x = torch.ones(1,2)*4
    X = torch.randn(1000,2)

    x, f, XX = gridSearch(x, X, N=8, niter=20, R=4)

    # for plotting
    fX = energyFunction(X)
    fXX = energyFunction(XX)
    plt.scatter(X[:,0],X[:,1], c=torch.log(fX+1e-4), alpha=0.2)
    plt.plot(XX[:,0],XX[:,1],'xr')

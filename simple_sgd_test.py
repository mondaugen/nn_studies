# Solve Least-Squares with SGD

# find min_x ((Ax - y)^T * (Ax - y))

from chainer import Variable
from numpy import *

gaus=random.standard_normal

# Number of datapoints
M=100
# Number of regressors
N=3

A=Variable(gaus((M,N)))
y=Variable(gaus((M,1)))

# Random starting point for gradient descent
x=Variable(gaus((N,1)))

# Loss function
L_x = (A @ x - y).T @ (A @ x - y)

# Learning rate

gam=1e-2
n_iter = 50

for n in n_iter:
    L_x.backward()
    x -= gam * x.grad
    print(L_x.data)

# Solve Least-Squares with SGD

# find min_x ((Ax - y)^T * (Ax - y))

from numpy import *

gaus=random.standard_normal

# Number of datapoints
M=100
# Number of regressors
N=3

A=gaus((M,N))
y=gaus((M,1))

# Random starting point for gradient descent
x=gaus((N,1))

# Loss function
def L_x(x):
    return (A @ x - y).T @ (A @ x - y)

# Learning rate

gam=1e-2
n_iter = 50

for n in range(n_iter):
    L_x.backward()
    grad = x.grad
    x.data -= gam * grad
    #L_x = (A @ x - y).T @ (A @ x - y)
    print(L_x.data)

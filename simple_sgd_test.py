# Solve Least-Squares with SGD

# find min_x ((Ax - y)^T * (Ax - y))

import chainer
from chainer import Variable
from numpy import *

import chainer
#import chainer.functions as F
from chainer import initializers
import numpy as np

gaus=random.standard_normal

class LinearRegression(chainer.Link):

    def __init__(self, n_in, n_out):
        super(LinearRegression, self).__init__()
        with self.init_scope():
            self.x = chainer.Parameter(
                    gaus((n_in,n_out)).astype('float32'), (n_in,n_out))
            #self.b = chainer.Parameter(
            #    gaus((n_out,1)).astype('float32'), (n_out,1))

    def __call__(self, a):
        # Call with a row of A
        y_ = a @ self.x# + self.b
        #return (y - y_) ** 2.
        return y_

# Number of datapoints
M=100
# Number of coefficients
N=3
# Number of output variables
O=1
# Learning rate
gam=1e-4

# Loss function
# L_x = (A @ x - y).T @ (A @ x - y)

lr=LinearRegression(N,O)
def L_x(x,t):
    y = lr(x)
    return (y - t).T @ (y - t)
sgd=chainer.optimizers.SGD(lr=gam).setup(lr)

#A=Variable(gaus((M,N)).astype('float32'))
#y=Variable(gaus((M,1)).astype('float32'))
A=gaus((M,N)).astype('float32')
y=gaus((M,1)).astype('float32')

n_iter = 500

for n in range(n_iter):
    idcs=list(range(A.shape[0]))
    #random.shuffle(idcs)
    for i,y_ in zip(idcs,y):
        sgd.update(L_x,A[i][None],y_.reshape((1,1)))
        #random.shuffle(idcs)
    e=A@lr.x.data - y
    print('Loss: %f' % (e.T@e,))

print('Gradient descent least squares')
print('Error')
e=A@lr.x.data - y
print(e.T@e)
print('coefficients')
print(lr.x.data)
print('Least squares loss')
print('Error')
x_,l,_,_=linalg.lstsq(A,y,rcond=None)
print(l)
print('coefficients')
print(x_)

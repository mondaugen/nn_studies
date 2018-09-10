# If W is used twice in one chain, what is the gradient?

import chainer
from chainer import Variable, functions
from numpy import *

gau=random.standard_normal

def sigmoid(x):
    return 1./(1.+exp(-x))

def printvar(name,x):
    print(name)
    print(x)
    print()

N = 2

W=gau((N,N))
x=gau((N,1))
u0=W@x
u1=W@u0
#du1_dW=ones((N,1))*u0.T + W@(ones((N,1))*x.T)
#du1_dW=ones((N,1))*u0.T + W.T@ones((N,1))*x.T
du1_dW=ones((N,1))*u0.T + W.T@ones((N,1))*x.T

printvar('theoretical du1_dW',du1_dW)

W=Variable(W)
x=Variable(x)
u0=W@x
u1=W@u0

u1.grad=ones((N,1))
u1.backward()
printvar('chainer\'s du1_dW',W.grad)

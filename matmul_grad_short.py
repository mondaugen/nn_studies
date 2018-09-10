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
u=W@x
du_dx=W.T@ones((N,1))
print(du_dx)

W=Variable(W)
x=Variable(x)
u=W@x
u.grad=ones((N,1))
u.backward()
print(x.grad)

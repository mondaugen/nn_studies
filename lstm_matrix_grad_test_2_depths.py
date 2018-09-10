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

# Theoretical
W=gau((N,N))
xt=gau((N,1))
ct1=gau((N,1))
a=gau((N,1))
f=gau((N,1))
o=gau((N,1))

i=W@xt
u7=W@ct1
u6=tanh(a)
u5=sigmoid(i)
u4=sigmoid(f)
u3=u4*u7
u2=u6*u5
u1=sigmoid(o)
ct=u2+u3
u0=tanh(ct)
h=u0*u1

dh_dW=u1*(1-tanh(ct)**2)*(u6*sigmoid(i)*(1-sigmoid(i))*xt.T
        + u4*ct1.T)

print('Theoretical:')
printvar('h',h)
printvar('dh_dW',dh_dW)

W=Variable(W)
xt=Variable(xt)
ct1=Variable(ct1)
a=Variable(a)
f=Variable(f)
o=Variable(o)

i=W@xt
u7=W@ct1
u6=functions.tanh(a)
u5=functions.sigmoid(i)
u4=functions.sigmoid(f)
u3=u4*u7
u2=u6*u5
u1=functions.sigmoid(o)
ct=u2+u3
u0=functions.tanh(ct)
h=u0*u1

h.grad=ones((N,1))
h.backward()
print('Chainer:')
printvar('h',h.data)
printvar('dh_dW',W.grad)

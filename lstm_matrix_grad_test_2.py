# now take two inputs to be a matrix-vector products, each using the same matrix
# and compute the gradient of the matrix

import chainer
from numpy import *

def sigmoid(x):
    return 1./(1.+exp(-x))

M = 3
N = 2


v=dict()
for k in ['a','f','o']:
    v[k]=random.standard_normal(N)

v['xt'] = random.standard_normal(M)
v['qt'] = random.standard_normal(M)
v['W'] = random.standard_normal((N,M))
v['i'] = v['W'] @ v['xt']
v['ct1'] = v['W'] @ v['qt']

u1=sigmoid(v['o'])
u6=tanh(v['a'])
u5=sigmoid(v['i'])
u4=sigmoid(v['f'])
u3=u4*v['ct1']
u2=u6*u5
ct=u2+u3
u0=tanh(ct)
h=u0*u1
dh_di=u1*(1-tanh(ct)**2)*u6*sigmoid(v['i'])*(1-sigmoid(v['i']))
dh_dct1=u1*(1-tanh(ct)**2)*1*u4
dh_dW=dh_di[None].T*v['xt'][None]+dh_dct1[None].T*v['qt'][None]

print('\nTheoretical gradient of dh/dW')
print('should be the same as chainer\'s dh/dW')
print(dh_dW)
print('\nTheoretical gradient of dh/di')
print('should be the same as chainer\'s dh/di')
print(dh_di)
print('\nTheoretical gradient of dh/dct1')
print('should be the same as chainer\'s dh/dct1')
print(dh_dct1)
print('\nTheoretical h')
print(h)
print('\nTheoretical ct')
print(ct)

# Now chainer
V=dict()

V['qt']=chainer.Variable(v['qt'][None].T)
V['W']=chainer.Variable(v['W'])
V['xt']=chainer.Variable(v['xt'][None].T)
V['i']=V['W'] @ V['xt']
V['ct1']=V['W'] @ V['qt']
#V['x']=chainer.Variable(concatenate((v['a'][None],v['i'][None],v['f'][None],v['o'][None])).T)
V['x']=chainer.functions.concat((v['a'][None].T,V['i'],v['f'][None].T,v['o'][None].T))
print(V['x'])

V['ct'],V['h']=chainer.functions.lstm(V['ct1'],V['x'])
print('\nChainer h')
print(V['h'])
print('\nChainer ct')
print(V['ct'])
V['h'].grad=ones((N,1))
V['h'].backward()
print('\nTrue Gradient of dh/dW')
print(V['W'].grad)
print('\nTrue Gradient of dh/di')
# TODO: Why is this None? Maybe chainer doesn't compute the gradients of
# intermediates?
print(V['i'].grad)
#print(V['x'].grad)
print('\nTrue Gradient of dh/ct1')
# This will also be None
print(V['ct1'].grad)

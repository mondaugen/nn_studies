# see if we understand what LSTM is computing

import chainer
from numpy import *

def sigmoid(x):
    return 1./(1.+exp(-x))


v=dict()
for k in ['a','i','f','ct1','o']:
    v[k]=random.standard_normal(2)


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

print('\nTheoretical gradient of dh/di')
print('should be the same as the 2nd column of chainer\'s dh/dx')
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

V['ct1']=chainer.Variable(v['ct1'][None].T)
print(V['ct1'])
V['x']=chainer.Variable(concatenate((v['a'][None],v['i'][None],v['f'][None],v['o'][None])).T)
print(V['x'])

V['ct'],V['h']=chainer.functions.lstm(V['ct1'],V['x'])
print('\nChainer h')
print(V['h'])
print('\nChainer ct')
print(V['ct'])
V['h'].grad=ones((2,1))
V['h'].backward()
print('\nTrue Gradient of dh/dx')
print(V['x'].grad)
print('\nTrue Gradient of dh/ct1')
print(V['ct1'].grad)

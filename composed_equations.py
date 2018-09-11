# Compose an equation and see if it gives you the right thing
# So this works
# You also get a nice recursive formula for the derivative of a matrix power
# w.r.t. the matrix
# If W is NxN
# d_dW(W^n@x) = ones((N,1))*(W^(n-1)@x)^T + W^T@d_dw(W^(n-1)@x)

import chainer
from chainer import Variable
from numpy import *

gaus=random.standard_normal

N=2

W=Variable(gaus((N,N)))
x=Variable(gaus((N,1)))

D=4

def Wx(x):
    return W @ x

y = x
for _ in range(D):
    y = Wx(y)

print(y.data)
print(linalg.matrix_power(W.data,D) @ x.data)

print("Theoretical gradient")
dy_dW=None
def D2():
    return ones((N,1))*(W.data@x.data).T \
            + W.data.T@ones((N,1))*x.data.T
def D3():
    return ones((N,1))*(linalg.matrix_power(W.data,2)@x.data).T \
            + W.data.T@D2()

if D == 1:
    dy_dW = ones((N,1))*x.data.T
if D == 2:
    dy_dW = D2()
if D == 3:
    dy_dW = D3()
if D == 4:
    dy_dW = ones((N,1))*(linalg.matrix_power(W.data,3)@x.data).T \
            + W.data.T@D3()

print(dy_dW)

y.grad = ones((N,1))
y.backward()
print("True gradient")
print(W.grad)

# This is not so necessary actually, defining a variable and using it multiple
# places can always be decomposed into a variable being used in multiple
# expressions, anyhow let's try.

from numpy import *
from chainer import Variable, functions

def sigmoid(x):
    return 1./(1.+exp(-x))

N=2
Wc=random.standard_normal((N,N))
x=random.standard_normal((N,1))
r=random.standard_normal((N,1))

u4 = Wc @ x
u3 = u4
u2 = u3*r
u1 = sigmoid(u4)
u0 = Wc @ u2
y = u1*u0

#dy_dWc = (u0 * sigmoid(u4)*(1 - sigmoid(u4)))[None].T*(ones((N,1))*x.T) + u1*(u2 + Wc @ (r*x))
#dy_dWc = (u0 * sigmoid(u4)*(1 - sigmoid(u4)))*x.T + u1*(ones((N,1))*u2.T + Wc.T @ ones((N,1)) * r * x.T)
dy_dWc = u0 * sigmoid(u4) * (1 - sigmoid(u4)) * (ones((N,1)) * x.T) \
        + u1 * ((ones((N,1)) * u2.T) + (Wc.T@ones((N,1))) * r * (ones((N,1)) \
        * x.T))
print('Theoretical y')
print(y)
print()

print('Theoretical gradient')
print(dy_dWc)
print()

Wc=Variable(Wc)
x=Variable(x)
r=Variable(r)

u4 = Wc @ x
u3 = u4
u2 = u3*r
u1 = functions.sigmoid(u4)
u0 = Wc @ u2
y = u1*u0

print('Chainer\'s y')
print(y)
print()

y.grad=ones((N,1))
y.backward()
print('Chainer\'s dy_dWc')
print(Wc.grad)
print()
print(Wc.grad - dy_dWc)

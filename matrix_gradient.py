import chainer
from numpy import *

Wc=random.standard_normal((2,3))
x=random.standard_normal((3,1))
Ru=random.standard_normal((2,1))
bc=random.standard_normal((2,1))

u1=Wc.dot(x)+bc
print('Ru*(1-tanh(u1)**2)')
print(Ru*(1-tanh(u1)**2))
print('ones((2,1))*x.T')
print(ones((2,1))*x.T)
print('theoretical gradient')
dy_dWc=(Ru*(1-tanh(u1)**2))*(ones((2,1))*x.T)
print(dy_dWc)

# Now let chainer do it
Wc_=chainer.Variable(Wc)
x_=chainer.Variable(x)
Ru_=chainer.Variable(Ru)
bc_=chainer.Variable(bc)
y=chainer.functions.tanh(Wc_ @ x_ + bc_) * Ru_
#y=Wc_ @ x_

y.grad=ones((2,1))
y.backward()
print('chainer gradient')
print(Wc_.grad)
print('should be same')

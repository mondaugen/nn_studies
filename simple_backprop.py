import chainer
from numpy import *

x0=random.random()
x1=random.random()

x=[chainer.Variable(array(x0)),chainer.Variable(array(x1))]

f=x[0]**2*(x[1]-1)

f.backward()

print(array([x[0].grad,x[1].grad]))
print(array([2*x0*(x1-1),x0*x0]))

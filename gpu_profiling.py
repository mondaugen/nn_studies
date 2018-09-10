import timeit
import cupy as cp
import numpy as np
import time

niter=3000
N=4096

X=np.random.standard_normal((N,N))
#Y=np.random.standard_normal((N,N))

time0=time.time()
for _ in range(niter):
    #X = X @ Y
    X = np.fft.fft(X)
time1=time.time()

print("np: %f" % (time1-time0))

with cp.cuda.Device(0):
    X=cp.random.standard_normal((N,N))
    Y=cp.random.standard_normal((N,N))

time0=time.time()
with cp.cuda.Device(0):
    for _ in range(niter):
        #X = X @ Y
        X = cp.fft.fft(X)
time1=time.time()

print("cp: %f" % (time1-time0))

import numpy as np
import matplotlib.pyplot as plt
import random

def find_k(N):
    return 1 + np.log(N/(2*(N-1)))/np.log((N-2)/N)

class Transposer:
    def __init__(self,i):
        """
        (Indexes start at 0)
        Swap 0th index and ith index, otherwise identity mapping.
        """
        self.i = i
    def map(self,x):
        if x == 0:
            return self.i
        if x == self.i:
            return 0
        return x

class RandomPermuter:
    def __init__(self,n_gens,n):
        """
        Picks n_gens integers in [0,n) to make transposers that, when composed,
        form a permutation. Of course if there are and even number of drawn
        integers that are equal, they cancel each other out. We could keep
        drawing numbers until we have n_gens unique numbers... but we don't for
        this demo.
        """
        gen_is = [random.randint(0,n-1) for _ in range(n_gens)]
        self.trans = [Transposer(g) for g in gen_is]
    def map(self,x):
        for t in self.trans:
            x = t.map(x)
        return x

# size of set to permute
N=101
# number of transpositions to compose
#k=find_k(N)
k=50
print("k: %d" % (k,))
k=int(k)
# number of trials
T=10000

# We run T trials where we run index 1 through the permuter. If it is
# transposed, we increment counter and then at the end divide by T to get
# probability of transposition

cnt = [0 for _ in range(N)]
for t in range(T):
    rp = RandomPermuter(k,N)
    for n in range(N):
        if rp.map(n) != n:
            cnt[n] += 1

def trans_prob(k,N):
    """
    Returns tuple, first is probability of 0th index being transposed, second is
    probability of nth index being transposed with n > 0
    """
    p = ((N-2)/N)**(k-1)*(N-1)/N
    return (p,1-p)

print(*[float(c)/T for c in cnt])
#print("predicted: %f" % ((1-1/N,)))
print("predicted: %f" % ((1-(1+3*(N-1))/(N**3),)))
print("trans prob: %e" % (sum(cnt)/float(T*N),))

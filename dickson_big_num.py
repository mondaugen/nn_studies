import random
import numpy as np
import sympy
from scipy.special import binom

def find_smallest_coprime(x):
    """
    find the smallest prime not in the list of factors
    """
    f=sympy.ntheory.factorint(x)
    n = 1
    while True:
        cf = sympy.ntheory.generate.prime(n) 
        if cf not in f.keys():
            ret = cf
            break
        n += 1
    return ret

def dn(n,x,a):
    # iterative
    ans = x
    prev_ans = 2
    if n == 1:
        return ans
    if n == 0:
        return prev_ans
    for i in range(2,n+1):
        tmp = ans
        ans = x * ans - a * prev_ans
        prev_ans = tmp
    return ans

Q = sympy.ntheory.nextprime(int(1e15))

n = find_smallest_coprime(Q*Q - 1)

cnt=0
T=10000

a = random.randint(1,Q)
for t in range(T):
    x = random.randint(0,Q-1)
    if dn(n,x,a) != x:
        cnt += 1

print("a: %d" % (a,))
print("Q: %d" % (Q,))
print("n: %d" % (n,))
print("trans prob: %e" % (float(cnt)/T),)

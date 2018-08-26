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

def d5(x,a):
    return x**5 - 5*x**3*a + 5*x*a**2

def d3(x,a):
    return x**3 - 3 * x * a

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

Q = sympy.ntheory.nextprime(100)

n = find_smallest_coprime(Q*Q - 1)

perms=[]
for a in range(Q):
    perms.append([])
    for x in range(Q):
        perms[-1].append(dn(n,x,a)%Q)
    perms[-1]=tuple(perms[-1])

if False:
    for p in perms:
        print(p)

print("Q: %d" % (Q,))
print("num unique permuations: %d"%(len(set(perms))))

print("n: %d" % (n,))

# Find probability of being transposed
cnt=0
T=len(perms)*Q
for p in perms:
    for i,x in enumerate(p):
        if i != x:
            cnt += 1

print("trans prob: %e" % (float(cnt)/T),)

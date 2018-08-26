import math
import random
import numpy as np
import sympy
from scipy.special import binom
import sys

# try and find all the permutations using a dickson polynomial

def find_next_smallest_coprime(x):
    """
    find the next smallest prime not in the list of factors
    """
    f=sympy.ntheory.factorint(x)
    n = 1
    while True:
        cf = sympy.ntheory.generate.prime(n) 
        if cf not in f.keys():
            yield cf
        n += 1

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

Q = sympy.ntheory.nextprime(10)

gn = find_next_smallest_coprime(Q*Q - 1)

perms = set()
n_perms = math.factorial(Q)

n = 1
while len(perms) < n_perms:
    try:
        n = gn.__next__()
        for a in range(Q):
            perm = []
            for x in range(Q):
                perm.append(dn(n,x,a)%Q)
            perms.add(tuple(perm))
    except KeyboardInterrupt:
        print(len(perms))
        sys.exit()

print("n: %d" % (n,))

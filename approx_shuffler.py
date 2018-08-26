import random
import numpy as np
import sympy
from scipy.special import binom

HASHER_LEN=3

x = random.randint(0,int(1e12))

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

def finite_power(x,p,q):
    ret = 1
    for _ in range(p):
        ret = (ret * x) % q

def dickson_eval(n,a,x,q):
    """
    evaluate nth Dickson polynomial of the first kind at x with parameter a in
    the finite field Fq
    """
    i=np.arange(int(np.floor(n/2)))
    #quo=(n*np.array([sympy.mod_inverse(n-i_,q) for i_ in i])) % q
    quo=n/(n-i)
    pows = n-2*i
    coefs=np.zeros(max(pows)+1)
    coefs[pows] = quo*binom(n-i,i)*np.power(-a,i)
    return sympy.polys.galoistools.gf_eval(coefs,x,q,sympy.polys.domains.ZZ)

class ShuffleMap(object):

    def __init__(self,Q,a=1):
        self.a = a
        self.n = find_smallest_coprime(Q*Q - 1)
        print(Q*Q - 1)
        print(self.n)
        self.Q = Q

    def map(self,x):
        """
        Maps x randomly onto range [0,Q) (hopefully)
        """
        return dickson_eval(self.n,self.a,x,self.Q)


def count_duplicates(a):
    return len(a) - len(set(a))

Q=11
sm = ShuffleMap(Q)
y=[sm.map(n) for n in range(Q)]
print("num duplicates: %d\n" % (count_duplicates(y)))
print(y)

#    gf_eval(f, a, p, K)
#            Evaluate ``f(a)`` in ``GF(p)`` using Horner scheme.
#
#                    Examples
#                            ========
#
#                                    >>> from sympy.polys.domains import ZZ
#                                            >>> from sympy.polys.galoistools
#                                            import gf_eval
#
#                                                    >>> gf_eval([3, 2, 4], 2, 5,
#                                                            ZZ)
#                                                            0
#
#

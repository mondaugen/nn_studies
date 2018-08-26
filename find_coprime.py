import sympy
import random


x = random.randint(0,int(1e12))
f=sympy.ntheory.factorint(x)

# now find the smallest prime not in the list of factors
n = 1
while True:
    cf = sympy.ntheory.generate.prime(n) 
    if cf not in f.keys():
        ret = cf
        break
    n += 1

print(ret)


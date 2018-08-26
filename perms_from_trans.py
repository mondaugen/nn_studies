import random

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

rp = RandomPermuter(10,10)

print(*[rp.map(x) for x in range(10)])

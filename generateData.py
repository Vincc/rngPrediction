import random
from numpy import array

def ohe(seq, setSize):
    outputseq = []
    for i in seq:
        temp = [0 for i in range(setSize+1)]
        temp[i-1] = 1
        outputseq.append(temp)
    return outputseq

#generates a set of seeded random intergers  of a certain size within the range(inclusive)
def generateSet( size, numrange):
    outputSet = []
    for i in range(size):
        outputSet.append(random.randint(numrange[0], numrange[1]))
    return outputSet

lcgSeed = random.randint(0,1000)
def lcg():
    a = 25096281518912105342191851917838718629
    c = 0
    m = 2**128
    global lcgSeed
    rand = (a*lcgSeed + c) % m
    return (rand/m)*10

for i in range(50):
    print(lcg())
    lcgSeed = lcg()
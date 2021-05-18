import random
from numpy import array

def ohe(seq, setSize):
    outputseq = []
    for i in seq:
        temp = [0 for i in range(setSize)]
        temp[i-1] = 1
        outputseq.append(temp)
    return outputseq

#generates a set of seeded random intergers  of a certain size within the range(inclusive)
def generateSet( size, numrange):
    outputSet = []
    for i in range(size):
        outputSet.append(random.randint(numrange[0], numrange[1]))
    #
    return array(ohe(outputSet, numrange[1]-numrange[0]+1))




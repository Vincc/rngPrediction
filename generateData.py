import random

def ohe(seq, setSize):
    outputseq = []
    for i in seq:
        temp = [0 for i in range(setSize)]
        temp[i-1] = 1
        outputseq.append(temp)
    return outputseq

#generates a set of seeded random intergers  of a certain size within the range(inclusive)
def generateSet(seed, size, numrange):
    outputSet = []
    random.seed(seed)
    for i in range(size):
        outputSet.append(random.randint(numrange[0], numrange[1]))
    return outputSet, ohe(outputSet, numrange[1]-numrange[0]+1)

print(generateSet(1,10,(1,10)))


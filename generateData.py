import random

#generates a set of seeded random intergers  of a certain size within the range(inclusive)
def generateSet(seed, size, numrange):
    outputSet = []
    random.seed(seed)
    for i in range(size):
        outputSet.append(random.randint(numrange[0], numrange[1]))
    return outputSet


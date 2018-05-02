import sys
import os
sys.path.append(os.path.relpath("/home/marcello/Documents/tec/ia/ia-pc1/codigo"))
from funciones4 import *

import decisionTree as dt

N = 5000
TESTPART = 0.9
LABELIDX = 0
doot = 0

# con N = 100 y TESTPART = 0.2
workPartRatio = 1 - TESTPART            # 0.8
trainPartRatio = 0.8 * workPartRatio    # 0.64
prunningSetRatio = 1 - trainPartRatio   # 0.36

workPartIdx = int(N * workPartRatio)             # 80
trainPartIdx = int(workPartIdx * trainPartRatio) # 64

# arr[:64]   = trainSet
# arr[64:80] = prunningSet
# arr[80:]   = testSet

def runAlgorithm():
    dataset = generar_muestra_pais(N)

    rawTree = dt.build_tree(dataset[:trainPartIdx])
    dt.print_tree(rawTree)

    #dt.prune_tree(rawTree, dataset[trainPart:workPartIdx])

    testing_data2 = dataset[workPartIdx:]
    errors = 0
    for row in testing_data2:
        guess = dt.print_leaf(dt.classify(row, rawTree))
        #global doot
        #print("beep: " + str(doot))
        #doot += 1
        for beep in guess.keys():
            if(row[0] != beep):
                errors += 1
                #print("Mistake on %s for %s" % (row[0], guess))

    testTotal = (N*(1-workPartRatio))
    print("Accuracy: " + str(100-((errors*100)/testTotal)))
    return 100-((errors*100)/testTotal)

if __name__ == '__main__':
    #results = []
    #for i in range(100):
    #    if i % 10 == 0:
    #        print("iter: " + str(i))
    #    results.append(runAlgorithm())

    #avg = 0
    #for i in results:
    #    avg += i
    #print("Average accuracy: " + str(avg/len(results)) + str("%"))

    runAlgorithm()

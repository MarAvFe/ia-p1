import sys
import os
sys.path.append(os.path.relpath("/home/marcello/Documents/tec/ia/ia-pc1/codigo"))
from funciones4 import *

import decisionTree as dt

N = 1000
trainPart = 0.2
LABELIDX = 1

def runAlgorithm():
    dataset = generar_muestra_pais(N)
    trainingTopIndex = int(N*trainPart)

    my_tree = dt.build_tree(dataset[:trainingTopIndex], LABELIDX)

    #dt.print_tree(my_tree)

    testing_data2 = dataset[trainingTopIndex:]

    errors = 0
    for row in testing_data2:
        #print ("Actual: %s. Predicted: %s" % (row[0], dt.print_leaf(dt.classify(row, my_tree))))
        pred = dt.print_leaf(dt.classify(row, my_tree))
        for beep in pred.keys():
            if(row[0] != beep):
                errors += 1
                #print("Mistake on %s for %s" % (row[0], pred))

    #print("Accuracy of %f" % (100-(N/errors)) + str("%"))
    return 100-(N/errors)

if __name__ == '__main__':
    results = []
    for i in range(100):
        results.append(runAlgorithm())

    avg = 0
    for i in results:
        avg += i
    print("Average accuracy: " + str(avg/len(results)) + str("%"))

# Next steps
# - add support for missing (or unseen) attributes
# - prune the tree to prevent overfitting
# - add support for regression

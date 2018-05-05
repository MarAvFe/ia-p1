import math
import numpy as np

def common_elements(list1, list2):
    return list(set(list1) & set(list2))

def knnByFoot(set, testRate, k):
    divideIdx = round(len(set)*(1-testRate))
    train = set[:divideIdx]
    test  = set[divideIdx:]
    guesses = []
    for i in range(len(test)):
        distances = []
        for j in range(len(train)):
            diff = 0
            for u in range(len(set[i])):
                diff += abs(set[i][u]-set[j][u])
            distances.append([diff, train[j]])
        guess = closest(distances, k)
        guesses.append(guess)
    accuracy = 100-compare(guesses, test)
    return accuracy, guesses

def closest(distances, k):
    sortDist = sortByDistance(distances)
    closest = sortDist[:k]
    tags = []
    for c in closest:
        tags.append(c[1][0])
    counts = []
    for t in common_elements(tags,tags):
        counts.append((t,tags.count(t)))
    tag, amount = "", 0
    for c in counts:
        if c[1] > amount:
            tag, amount = c[0], c[1]
    return tag

def sortByDistance(arr):
    # sorts an array no the form [ int, [list] ]
    for i in range(1,len(arr)):
        j = i
        while j > 0 and arr[j] < arr[j-1]:
            arr[j], arr[j-1] = arr[j-1], arr[j]
            j = j-1
    return arr

def compare(guesses, testSet):
    total = len(guesses)
    errors = 0
    for i in range(total):
        if guesses[i] != testSet[i][0] :
            errors += 1
    return errors*100/total

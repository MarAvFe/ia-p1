# For Python 2 / 3 compatability
from __future__ import print_function
import math

LABELIDX = 0

class Condition:
    def __init__(self, index, value):
        self.index = index
        self.value = value

    def cmp(self, example):
        val = example[self.index]
        if (isinstance(val, bool)):
            return val
        elif isinstance(val, int) or isinstance(val, float):
            return val >= self.value
        elif val == None:
            return False
        else:
            return val == self.value

    def __repr__(self):
        condition = "=="
        if (isinstance(self.value, bool)):
            condition = "=="
        elif isinstance(self.value, int) or isinstance(self.value, float):
            condition = ">="
        elif self.value == None:
            condition = "¿?"
        return "Es %s %s %s?" % (
            header[self.index], condition, str(self.value))

def entropy(data):
    #https://datascience.stackexchange.com/questions/10228/gini-impurity-vs-entropy#10273
    #https://en.wikipedia.org/wiki/Decision_tree_learning#Implementations
    total = class_counts(data)
    uncertainty = 1
    for tag in total:
        percentage = total[tag] / float(len(data))
        uncertainty -= percentage * math.log(percentage,10)
    return uncertainty


header = ["primerVoto", "segundoVoto", "canton", "sexo", "edad", "densidadPoblación", "dependienteEconom", "promOcupantes", "estadoVivienda", "viveEnEacinamiento", "promAlfabetismo", "añosAprobadosEducacion", "porcentAsistEducRegular", "estaDesempleado", "activoEconomicamente", "estaDesempleado", "estaAsegurado", "nacidoExtranjero", "esDiscapacitado", "hogarJefaturaCompartida"]


def class_counts(rows):
    counts = {}  # a dictionary of label -> count.
    for row in rows:
        # in our dataset format, the label is always the last column
        label = row[LABELIDX]
        if label not in counts:
            counts[label] = 0
        counts[label] += 1
    return counts


def partition(rows, question):
    true_rows, false_rows = [], []
    for row in rows:
        if question.cmp(row):
            true_rows.append(row)
        else:
            false_rows.append(row)
    return true_rows, false_rows


def info_gain(left, right, current_uncertainty):
    p = float(len(left)) / (len(left) + len(right))
    return current_uncertainty - p * entropy(left) - (1 - p) * entropy(right)


def find_best_split(rows):
    best_gain = 0  # keep track of the best information gain
    best_condition = None  # keep train of the feature / value that produced it
    current_uncertainty = entropy(rows)
    n_features = len(rows[0]) - 1  # number of columns
    for col in range(n_features):  # for each feature

        values = set([row[col] for row in rows])  # unique values in the column

        for val in values:  # for each value

            condition = Condition(col, val)

            true_rows, false_rows = partition(rows, condition)

            if len(true_rows) == 0 or len(false_rows) == 0:
                # pure partition
                continue

            gain = info_gain(true_rows, false_rows, current_uncertainty)
            if gain >= best_gain:
                best_gain, best_condition = gain, condition

    return best_gain, best_condition


class Leaf:
    def __init__(self, rows):
        self.guesses = class_counts(rows)
        self.rows = rows


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows):
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(tree, prefix="", depth=-1):
    if isinstance(tree, Leaf):
        print (prefix + "Total", tree.guesses)
        return
    print (prefix + str(tree.question) + str(depth))

    if depth == 0:
        print("--depth reached--")
        return

    print (prefix + '├─> True:')
    print_tree(tree.true_branch, prefix + "|  ", -1 if depth == -1 else depth-1)
    print (prefix + '└─> False:')
    print_tree(tree.false_branch, prefix + "  ", -1 if depth == -1 else depth-1)


def classify(row, node):
    if isinstance(node, Leaf):
        return node.guesses

    if node.question.cmp(row):
        return classify(row, node.true_branch)
    else:
        return classify(row, node.false_branch)


def print_leaf(counts):
    total = sum(counts.values()) * 1.0
    probs = {}
    for lbl in counts.keys():
        probs[lbl] = str(int(counts[lbl] / total * 100)) + "%"
    return probs

def get_node_rows(node):
    if isinstance(node, Leaf):
        return node.rows
    rows = [] + get_node_rows(node.true_branch) + get_node_rows(node.false_branch)
    return rows

def count_nodes(tree):
    if isinstance(tree, Leaf):
        return 1
    return count_nodes(tree.true_branch) + count_nodes(tree.false_branch) + 1

def prune_tree(node,rows):
    prune_threshold = 0.2
    if isinstance(node, Leaf):
        return node

    node.true_branch = prune_tree(node.true_branch, rows)
    node.false_branch = prune_tree(node.false_branch, rows)

    all_rows = get_node_rows(node)
    curEnt = entropy(all_rows)
    childEnt = (
                entropy(get_node_rows(node.true_branch)) +
                entropy(get_node_rows(node.false_branch))
                ) / 2

    if abs(curEnt - childEnt) < prune_threshold:
        return Leaf(all_rows)
    return node



def runAlgorithm(dataset, testRate):
    N = len(dataset)
    LABELIDX = 0

    # con N = 100 y testRate = 0.2
    workPartRatio = 1 - testRate            # 0.8
    trainPartRatio = 0.8 * workPartRatio    # 0.64
    prunningSetRatio = 1 - trainPartRatio   # 0.36

    workPartIdx = int(N * workPartRatio)             # 80
    trainPartIdx = int(workPartIdx * trainPartRatio) # 64

    # arr[:64]   = trainSet
    # arr[64:80] = prunningSet
    # arr[80:]   = testSet
    #dataset = generar_muestra_pais(N)

    accuracy, newTree = run_tree(dataset[:workPartIdx])
    print("nodos pre poda: " + str(count_nodes(newTree)))
    prunnedTree = prune_tree(newTree, dataset[workPartIdx:])
    #print_tree(prunnedTree, "")
    prunnedAccuracy = test_tree(prunnedTree, dataset[:workPartIdx])
    print("nodos post poda: " + str(count_nodes(newTree)))
    return 100 - prunnedAccuracy, []

def test_tree(tree, data):
    errors = 0
    for row in data:
        guess = print_leaf(classify(row, tree))
        morePlausibleGuessInGuessesLongVariableNameNotFunnyAnymore = (-1,0)
        for key in guess.keys():
            val = int(guess[key].split("%")[0])
            if (val > morePlausibleGuessInGuessesLongVariableNameNotFunnyAnymore[1]):
                morePlausibleGuessInGuessesLongVariableNameNotFunnyAnymore = (key, val)
        if(row[0] != morePlausibleGuessInGuessesLongVariableNameNotFunnyAnymore[0]):
            errors += 1
    testTotal = len(data)
    return 100-((errors*100)/testTotal)

def run_tree(dataset):
    splitIdx = int(len(dataset)*0.8)
    rawTree = build_tree(dataset[:splitIdx])
    #print_tree(rawTree, "")
    testing_data2 = dataset[splitIdx:]
    err = test_tree(rawTree, testing_data2)
    return err, rawTree

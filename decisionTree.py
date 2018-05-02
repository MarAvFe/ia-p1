#https://www.youtube.com/watch?v=LDRbO9a6XPU

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


class Decision_Node:
    def __init__(self,
                 question,
                 true_branch,
                 false_branch):
        self.question = question
        self.true_branch = true_branch
        self.false_branch = false_branch


def build_tree(rows, lblidx=0):
    global LABELIDX
    LABELIDX = lblidx
    gain, question = find_best_split(rows)

    if gain == 0:
        return Leaf(rows)

    true_rows, false_rows = partition(rows, question)

    true_branch = build_tree(true_rows)
    false_branch = build_tree(false_rows)

    return Decision_Node(question, true_branch, false_branch)


def print_tree(tree, prefix=""):
    if isinstance(tree, Leaf):
        print (prefix + "Total", tree.guesses)
        return
    print (prefix + str(tree.question))

    print (prefix + '├─> True:')
    print_tree(tree.true_branch, prefix + "|  ")
    print (prefix + '└─> False:')
    print_tree(tree.false_branch, prefix + "  ")


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

def prune_tree(node, rows):
    pass

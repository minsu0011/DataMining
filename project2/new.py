import sys
import math
from collections import Counter, defaultdict

class DecisionNode:
    def __init__(self, attribute=None, branches=None, label=None):
        self.attribute = attribute      
        self.branches = branches or {}  
        self.label = label              

def readDataset(filename, skipHeader=True):
    records = []
    with open(filename, 'r') as f:
        if skipHeader:
            next(f)
        for line in f:
            parts = line.strip().split('\t')
            if parts:
                records.append(parts)
    return records

def writeResults(headers, data, predictions, outputFile):
    with open(outputFile, 'w') as f:
        f.write('\t'.join(headers) + '\n')
        for record, label in zip(data, predictions):
            f.write('\t'.join(record + [label]) + '\n')

def entropy(labels):
    total = len(labels)
    counts = Counter(labels)
    ent = 0.0

    for count in counts.values():
        p = count / total
        ent -= p * (0 if p == 0 else math.log2(p))
    
    return ent

def gainRatio(data, labels, attrIdx):
    baseEnt = entropy(labels)
    total = len(labels)
    subsets = defaultdict(lambda: {'records': [], 'labels': []})

    for record, label in zip(data, labels):
        key = record[attrIdx]
        subsets[key]['records'].append(record)
        subsets[key]['labels'].append(label)

    condEnt = 0.0
    splitInfo = 0.0

    for subset in subsets.values():
        size = len(subset['labels'])
        p = size / total
        condEnt += p * entropy(subset['labels'])
        splitInfo -= p * (0 if p == 0 else math.log2(p))

    infoGain = baseEnt - condEnt
    if splitInfo == 0:
        return 0
    else : 
        return infoGain / splitInfo

def buildTree(data, labels, attributes):
    if len(set(labels)) == 1:
        return DecisionNode(label=labels[0])
    
    if not attributes:
        maj = Counter(labels).most_common(1)[0][0]
        return DecisionNode(label=maj)
    
    gains = [(gainRatio(data, labels, idx), idx) for idx in attributes]
    bestGain, bestAttr = max(gains, key=lambda x: x[0])

    if bestGain == 0:
        maj = Counter(labels).most_common(1)[0][0]
        return DecisionNode(label=maj)
    
    node = DecisionNode(attribute=bestAttr)
    subsets = defaultdict(lambda: {'records': [], 'labels': []})
    
    for record, label in zip(data, labels):
        key = record[bestAttr]
        subsets[key]['records'].append(record)
        subsets[key]['labels'].append(label)

    for attrVal, subset in subsets.items():
        childAttrs = [a for a in attributes if a != bestAttr]
        child = buildTree(subset['records'], subset['labels'], childAttrs)
        node.branches[attrVal] = child

    return node

def classify(node, record):
    if node.label is not None:
        return node.label
    
    val = record[node.attribute]

    if val in node.branches:
        return classify(node.branches[val], record)
    
    stack = [node]
    labels = []

    while stack:
        cur = stack.pop()
        if cur.label is not None:
            labels.append(cur.label)
        else:
            for child in cur.branches.values():
                stack.append(child)
    return Counter(labels).most_common(1)[0][0]

def main():
    trainFile, testFile, outputFile = sys.argv[1], sys.argv[2], sys.argv[3]

    with open(trainFile) as f:
        trainHeader = f.readline().strip().split('\t')
    with open(testFile) as f:
        testHeader = f.readline().strip().split('\t')

    trainData = readDataset(trainFile, skipHeader=True)  # 각 행: [age, income, …, Class:buys_computer]
    testData = readDataset(testFile,  skipHeader=True)  # 각 행: [age, income, …]

    attributes = list(range(len(trainData[0]) - 1))
    trainLabels = [row[-1] for row in trainData]
    trainRecords = [row[:-1] for row in trainData]

    tree = buildTree(trainRecords, trainLabels, attributes)
    predictions = [classify(tree, row) for row in testData]

    labelHeader = trainHeader[-1]
    headers = testHeader + [labelHeader]
    writeResults(headers, testData, predictions, outputFile)

if __name__ == "__main__":
    main()
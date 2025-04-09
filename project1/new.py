import sys
from itertools import combinations


def main() :
    minSupport = float(sys.argv[1])
    inputFileName = sys.argv[2]
    outputFileName = sys.argv[3]
    minSupport = minSupport / 100.0

    # Read the input file and parse it into transactions
    transactions = []
    with open(inputFileName , 'r') as f:
        for line in f:
            transaction = list(map(int, line.strip().split()))
            transactions.append(transaction)
    
    #Perform Apriori algorithm
    frequentItemsets = apriori(transactions, minSupport)

    # Generate associative rules from the frequent itemsets
    rules = generateRules(frequentItemsets)

    # Write the associative rules to the output file
    with open(outputFileName, 'w') as f:
        for subsetA, subsetB, support, confidence in rules:
            A_str = '{' + ','.join(map(str, sorted(subsetA))) + '}'
            B_str = '{' + ','.join(map(str, sorted(subsetB))) + '}'
            f.write(f"{A_str}\t{B_str}\t{support:.2f}\t{confidence:.2f}\n")




def apriori(transactions, minSupport):
    frequentItemsets = {}
    # Create and Compose the initial candidate C1 (Dictionary)
    C1 = {}
    L = {}
    prunedSet = set()
    for transaction in transactions:
        for item in transaction:
            if item not in C1:
                C1[item] = 0
            C1[item] += 1

    # Create and Compose the initial frequent patterns L
    for item, count in C1.items():
        if count / len(transactions) >= minSupport:
            L[frozenset([item])] = count / len(transactions)
        
        # Add the pruned itemsets to the prunedSet
        else :
            prunedSet.add(frozenset([item]))
    
    # Add the frequent patterns in L1 to the frequentItemsets
    for itemset, support in L.items():
            frequentItemsets[itemset] = support
    
    # Sort the frequent patterns L1 in ascending order(optional)
    L = dict(sorted(L.items(), key=lambda item: next(iter(item[0]))))


    # C2, l2, C3, l3, ... until no more frequent patterns are found
    while(True):
        
        # Reset Ck
        Ck = {}
        
        # Create and Compose the candidate Ck by self-joining L
        for item1 in L.keys():
            for item2 in L.keys():
                if item1 == item2:
                    continue
                candidate = item1.union(item2)
                # Check if candidate is a subset of prunedSet(Before pruning)
                shouldPrune = any(pruned <= candidate for pruned in prunedSet)
                if len(candidate) == len(item1) + 1 and not shouldPrune:
                    Ck[candidate] = 0
        
        # If no more Ck is found, break the Loop
        if(len(Ck) == 0) :
            break

        # Count the support(frequency) of Ck
        for transaction in transactions:
            t_set = set(transaction)
            for candidate in Ck.keys():
                if candidate <= t_set:
                    Ck[candidate] += 1
        
        # Create and Compose the frequent patterns Lk (Dictionary)
        L = {}
        newPrunedSet = set()
        for candidate, count in Ck.items():
            if count / len(transactions) >= minSupport:
                L[candidate] = count / len(transactions)

            # Add the prunedSet to the newPrunedSet
            else :
                newPrunedSet.add(candidate)

        #If no more L is found, break the Loop
        if(len(L) == 0) :
            break  

        # Add the frequent patterns Lk to the frequentItemsets
        for itemset, support in L.items():
            frequentItemsets[itemset] = support

        prunedSet = newPrunedSet

    return frequentItemsets
        

def generateRules(frequentItemsets):
    rules = []
    
    # Create and Compose the association rules(A -> B) from the frequent itemsets
    for itemset in frequentItemsets.keys():

        if len(itemset) < 2:
            continue

        for i in range(1, len(itemset)):
            for subsetA in combinations(itemset, i):
                
                subsetA = frozenset(subsetA)
                subsetB = itemset - subsetA

                # Check subsetA is frequent itemset
                if subsetA in frequentItemsets:
                    support = frequentItemsets[itemset] * 100
                    confidence = (frequentItemsets[itemset] / frequentItemsets[subsetA]) * 100
                    rules.append((subsetA, subsetB, round(support, 2), round(confidence, 2)))
    return rules


    

if __name__ == "__main__":
    main()
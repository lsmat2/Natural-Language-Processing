import csv
import numpy as np

# – Vector = 0-1 bit vector (word presence/absence) : vectorSpaceBitVector(documents, numDocs, query)
# – Similarity = dot product
# – f(q,d) = number of distinct query words matched in d

numEntries = 7000
trainFilePath = "train.csv"
testFilePath = "test.csv"
punctuation = [".", ",", ":", "'", ")", "(", "?", "-", "!"]
printStatementsOn = False

# HELPER FUNCTIONS
def remove_punctuation(inputString:str, charactersToRemove:list[str]) -> str:
    for char in charactersToRemove:
        inputString = inputString.replace(char, "")
    return inputString

def normalizedInputArray(input:str) -> list[str]:
    input = input.strip().lower()
    input = remove_punctuation(input, punctuation)
    return input.split(" ")

def readEntries(csvreader, numEntries):
    entries = []
    count = 0
    for entry in csvreader:
        entries.append(entry)
        if count == numEntries: break  # COUNTER FOR HOW MANY ENTRIES TO ANALYZE
        count += 1
    return entries

def getDataFromFile(filePath:str, numEntries:int) -> list:
    file = open(filePath)
    csvreader = csv.reader(file)
    data = readEntries(csvreader, numEntries)
    file.close()
    return data

def printTopFive(VSMmodel:list, data): # PRINT RESULT OF TOP-5 AND BOTTOM-5 MOST RELEVANT DOCUMENTS WITH RANKING VALUES
    # Get Index Array of Top 5 rankings
    indicesOfLargestFive = np.argsort(VSMmodel)[-5:] # Selects last 5 indices from ascending order array
    
    print("TOP FIVE RANKED DOCUMENTS:")
    for i in range(len(indicesOfLargestFive)):
        index = indicesOfLargestFive[len(indicesOfLargestFive) - 1 - i]
        score = VSMmodel[index]
        title = data[index][1]
        description = data[index][2]
        print("Doc Rank:", i + 1, "\nScore:", score, "\tTitle:", title, "\tDescription:", description)
    print("")

def printBottomFive(VSMmodel:list, data):
    # Get Index Array of Bottom 5 (nonzero) rankings
    VSMarray = np.array(VSMmodel)
    nonZeroIndices = np.nonzero(VSMarray)[0]
    numRankedDocuments = len(nonZeroIndices)
    sorted_indices = np.argsort(VSMarray[nonZeroIndices])
    filtered_indices = nonZeroIndices[sorted_indices]
    indicesOfSmallestFive = filtered_indices[:5]
    
    print("BOTTOM FIVE RANKED DOCUMENTS (>0):")
    for i in range(len(indicesOfSmallestFive)):
        index = indicesOfSmallestFive[len(indicesOfSmallestFive) - 1 - i]
        score = VSMmodel[index]
        title = data[index][1]
        description = data[index][2]
        print("Doc Rank:", numRankedDocuments - (3 - i), "\nScore:", score, "\tTitle:", title, "\tDescription:", description)
    print("")

def printTopAndBottomFive(VSMmodel:list, filePath:str):
    data = getDataFromFile(filePath, numEntries)
    printTopFive(VSMmodel, data)
    printBottomFive(VSMmodel, data)

# 1. Create a script that automates the process of computing word relevances using a vector space 
# representation using the bit-vector representation - without document length normalization.

# Document Relevance with Vector Space Basic Model 
def vectorSpaceBitVector(filePath:str, numEntries:int, query:str) -> list[int]:
    vectorSpaceScores = [0] * (numEntries + 1)
    # Retrieve data from file
    data = getDataFromFile(filePath, numEntries)
    # Normalize query
    normalizedQuery = normalizedInputArray(query)
    if printStatementsOn: print("Query words: ", normalizedQuery)
    # Calculate vector space score for each document
    index = 0
    for row in data:
        count = 0
        # Normalize title/description
        # normalizedTitle = normalizedInputArray(row[1])
        normalizedDesc = normalizedInputArray(row[2])
        # Count += 1 : For each time a query word appears in doc title/desc
        
        # Method 1: ignore duplicates
        # for word in normalizedTitle:
        #     if word in normalizedQuery: count += 1
        for word in normalizedDesc:
            if word in normalizedQuery: count += 1
        
        # Method 2: account for duplicates --> improved (Term Frequency Weighting)
        # for word in normalizedTitle:
        #     for queryWord in normalizedQuery: 
        #         if word == queryWord: count += 1
        # for word in normalizedDesc:
        #     for queryWord in normalizedQuery:
        #         if word == queryWord: count += 1
            
        # Update vector space model array
        vectorSpaceScores[index] = count
        index += 1
    
    if printStatementsOn:
        docID = 0
        for num in vectorSpaceScores:
            if num > 0: print("Document", docID, ":\t✅", num)
            else: print("Document", docID, ":\t", num)
            docID += 1

    return vectorSpaceScores


# 2. Test your implementation for the following queries:
#   • q = “olympic gold athens”
#   • q = “reuters stocks friday”
#   • q = “investment market prices”
query1 = "olympic gold athens"
query2 = "reuters stocks friday"
query3 = "investment market prices"

trainVSBV = vectorSpaceBitVector(trainFilePath, numEntries, query1)
trainVSBV2 = vectorSpaceBitVector(trainFilePath, numEntries, query2)
trainVSBV3 = vectorSpaceBitVector(trainFilePath, numEntries, query3)

# 3. Test your implementation for words from the test-set in the dataset.

if numEntries > 7600: numEntries = 7600
testVSBV = vectorSpaceBitVector(testFilePath, numEntries, query1)
testVSBV2 = vectorSpaceBitVector(testFilePath, numEntries, query2)
testVSBV3 = vectorSpaceBitVector(testFilePath, numEntries, query3)

# Print out results

print("\nQUERY: olympic gold athens\n")
print("Training:")
printTopAndBottomFive(trainVSBV, trainFilePath)
print("Test:")
printTopAndBottomFive(testVSBV, testFilePath)
print("\n\nQUERY: reuters stocks friday\n")
print("Train:")
printTopAndBottomFive(trainVSBV2, trainFilePath)
print("Test:")
printTopAndBottomFive(testVSBV2, testFilePath)
print("\n\nQUERY: investment market prices\n")
print("Train:")
printTopAndBottomFive(trainVSBV3, trainFilePath)
print("Test:")
printTopAndBottomFive(testVSBV3, testFilePath)
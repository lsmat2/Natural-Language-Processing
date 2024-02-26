import csv
# import pandas as pd



numEntries = 110
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
        normalizedTitle = normalizedInputArray(row[1])
        normalizedDesc = normalizedInputArray(row[2])
        # Count += 1 : For each time a query word appears in doc title/desc
        
        # Method 1: ignore duplicates
        for word in normalizedTitle:
            if word in normalizedQuery: count += 1
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
testVSBV2 = vectorSpaceBitVector(trainFilePath, numEntries, query2)
trainVSBV3 = vectorSpaceBitVector(trainFilePath, numEntries, query3)

# 3. Test your implementation for words from the test-set in the dataset.

testVSBV = vectorSpaceBitVector(testFilePath, numEntries, query1)
testVSBV2 = vectorSpaceBitVector(testFilePath, numEntries, query2)
testVSBV3 = vectorSpaceBitVector(testFilePath, numEntries, query3)
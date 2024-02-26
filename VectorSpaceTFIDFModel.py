import csv
import math


# – Vector = 0-1 bit vector (word presence/absence) : vectorSpaceBitVector(documents, numDocs, query)
# – Similarity = dot product
# – f(q,d) = number of distinct query words matched in d


numEntries = 1000
trainFilePath = "train.csv"
testFilePath = "test.csv"
punctuation = [".", ",", ":", "'", ")", "(", "?", "-", "!"]
printStatementsOn = False
BM25_k = 25

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

def docFrequency(queryTerm:str, documentList:list) -> float:
    numDocs = 0
    queryFrequency = 0
    for document in documentList:
        if queryTerm in normalizedInputArray(document[1]): queryFrequency += 1 # Checks for term in title
        elif queryTerm in normalizedInputArray(document[2]): queryFrequency += 1 # Checks for term in description
        numDocs += 1
    return float(queryFrequency / numDocs)

def numDocsInList(documentList:list) -> int:
    numDocs = 0
    for document in documentList : numDocs += 1
    return numDocs

def IDFweight(numDocs:int, frequency:float) -> float:
    if frequency == 0: return 0
    else: return math.log(float(numDocs + 1) / frequency)

def numStringMatches(input:str, stringList:list[str]) -> int:
    count = 0
    for string in stringList:
        if string == input: count += 1
    return count

def BM25(countWordInDoc:int, k: int) -> float: return (((k + 1)*countWordInDoc) / (countWordInDoc + k))

# 1. Create a script that automates the process of computing word relevances using a vector space
# representation using the TF-IDF using Okapi-BM25 - without document length normalization.

# Document Relevance with Vector Space TF-IDF Model 
def vectorSpaceIDFVector(filePath:str, numEntries:int, query:str) -> list[float]:
    vectorSpaceScores = [0] * (numEntries + 1)
    
    # Retrieve data from file
    data = getDataFromFile(filePath, numEntries)
    
    # Normalize query
    normalizedQuery = normalizedInputArray(query)
    # if printStatementsOn: print("Query words: ", normalizedQuery)
    
    # Initialize Frequency Map [query -> frequency] for each Query Term, and numDocs
    numDocs = numDocsInList(data)
    queryTermFrequencyList = {}
    queryCountInQueryList = {}
    for queryTerm in normalizedQuery:
        frequency = docFrequency(queryTerm, data)
        queryTermFrequencyList[queryTerm] = frequency
        queryCountInQueryList[queryTerm] = (numStringMatches(queryTerm, normalizedQuery))
    # Calculate vector space TF-IDF score for each document
    index = 0
    for document in data:
        documentScore = 0
        # Normalize title/description
        normalizedDesc = normalizedInputArray(document[2])

        # Method 3: TF weighting and IDF (inverse document frequency)
        # for queryWord in normalizedQuery:
        #     # Calculate IDF Weight [log(M+1)/df(w)]
        #     frequency = queryTermFrequencyList[queryWord]
        #     IDFweight = math.log(float(numDocs + 1) / frequency)
        #     # Calculate CountWordInQuery & CountWordInDoc
        #     countWordInQuery = queryCountInQueryList[queryWord]
        #     countWordInDoc = numStringMatches(queryWord, normalizedDesc)
        #     # Add current weight to document score 
        #     currentQueryWordWeight = countWordInQuery*countWordInDoc*IDFweight
        #     documentScore += currentQueryWordWeight

        # Method 4: TF Transformation: BM25 -->     (k + 1)x / (x + k)   --> choosing k = 25
        for queryWord in normalizedQuery:
            # Calculate IDF Weight [log(M+1)/df(w)]
            frequency = queryTermFrequencyList[queryWord]
            currIDFweight = IDFweight(numDocs, frequency)
            # Calculate CountWordInQuery & BM25_value (replaces CountWordInDoc)
            countWordInQuery = queryCountInQueryList[queryWord]
            countWordInDoc = numStringMatches(queryWord, normalizedDesc)
            BM25_Value = BM25(countWordInDoc, BM25_k)
            # Add current weight to document score
            currentTermScore = float(countWordInQuery) * BM25_Value * currIDFweight
            documentScore += currentTermScore
                            
        # Update vector space model array
        vectorSpaceScores[index] = documentScore
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

trainVSBV = vectorSpaceIDFVector(trainFilePath, numEntries, query1)
testVSBV2 = vectorSpaceIDFVector(trainFilePath, numEntries, query2)
trainVSBV3 = vectorSpaceIDFVector(trainFilePath, numEntries, query3)

# 3. Test your implementation for words from the test-set in the dataset.

if numEntries > 7600: numEntries = 7600
testVSBV = vectorSpaceIDFVector(testFilePath, numEntries, query1)
testVSBV2 = vectorSpaceIDFVector(testFilePath, numEntries, query2)
testVSBV3 = vectorSpaceIDFVector(testFilePath, numEntries, query3)
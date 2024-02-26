import csv


# – Vector = 0-1 bit vector (word presence/absence) : vectorSpaceBitVector(documents, numDocs, query)
# – Similarity = dot product
# – f(q,d) = number of distinct query words matched in d


numEntries = 100
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
        # Method: Term Frequency Weighting (account for duplicates)
        for word in normalizedTitle:
            for queryWord in normalizedQuery: 
                if word == queryWord: count += 1
        for word in normalizedDesc:
            for queryWord in normalizedQuery:
                if word == queryWord: count += 1
        # Update vector space model array
        vectorSpaceScores[index] = count
        index += 1

    return vectorSpaceScores

# 1. Create a script that automates the process of computing word relevances using a vector space 
# representation using the bit-vector representation - without document length normalization.

# Document Relevance with Vector Space Basic Model 
def vectorSpaceIDFVector(filePath:str, numEntries:int, query:str) -> list[float]:
    vectorSpaceScores = [0] * (numEntries + 1)
    # Retrieve data from file
    data = getDataFromFile(filePath, numEntries)
    # Normalize query
    normalizedQuery = normalizedInputArray(query)
    if printStatementsOn: print("Query words: ", normalizedQuery)
    # Initialize Frequency List for each Query Term
    queryTermFrequencyList = []
    for queryTerm in normalizedQuery:
        frequency = docFrequency(queryTerm, data)
        queryTermFrequencyList.append(frequency)
    # Calculate vector space TF-IDF score for each document
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

def docFrequency(queryTerm:str, documentList:list) -> float:
    numDocs = 0
    queryFrequency = 0
    for document in documentList:
        if queryTerm in normalizedInputArray(document[1]): queryFrequency += 1 # Checks for term in title
        elif queryTerm in normalizedInputArray(document[2]): queryFrequency += 1 # Checks for term in description
        numDocs += 1
    return float(queryFrequency / numDocs)



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

testVSBV = vectorSpaceIDFVector(testFilePath, numEntries, query1)
testVSBV2 = vectorSpaceIDFVector(testFilePath, numEntries, query2)
testVSBV3 = vectorSpaceIDFVector(testFilePath, numEntries, query3)
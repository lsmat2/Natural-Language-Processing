import csv
import pandas as pd
# from collections import Counter
import enchant

# import nltk
from nltk.corpus import stopwords
# nltk.download('stopwords')
# print(stopwords.words('english'))


# INSTRUCTIONS (PROBLEM 1)

# For this exercise, we will use the dataset above. 
# Create a function to read the text of the corpus. 
# Remove stop-words, transform words to lower-case, remove punctuation and numbers, 
    # and excess whitespaces (keeping only one space that separates words). 

# Create a vocabulary (a list of words) using the top 200 most frequently in the text. 
# Sort them in decreasing order of frequency. 
# The size of the dataset (training set) is large. 
# Thus, select a smaller subset (of 10,000 to 15,000) rows to develop your solution. 
# Feel free to use any library to implement the data access and parsing.


# Globals
printStatementsOn = False
numEntries = 30
# (Title Only)
# 200: ~4.2s
# 1000: ~12.4s
# 5000: ~53.0s
# (Description Included)
# 30: ~3.7s
# 200: ~14.3s
# 10000: ..... long time :(
punctuation = [".", ",", ":", "'", ")", "(", "?", "-"]
word_validator = enchant.Dict("en_US")
normalizedData = {}
termFreqDict = {} # [id] -> [ map(word -> count) ]
termFreqOverall = {} # [words] -> [counts across all documents]


# Removes [charactersToRemove] from [inputString]
def remove_punctuation(inputString:str, charactersToRemove:list[str]):
    for char in charactersToRemove:
        inputString = inputString.replace(char, "")
    return inputString

# Stores [Class, Title, Description] of [numEntries] entries into array of entries using csv reader
def readEntries(csvreader, numEntries):
    entries = []
    count = 0
    for entry in csvreader:
        entries.append(entry)
        if count == numEntries: break  # COUNTER FOR HOW MANY ENTRIES TO ANALYZE
        count += 1
    return entries

# 1) Retrieve data & initialize csv reader
file = open("train.csv")
csvreader = csv.reader(file)

# 2) Store [Class, Title, Description] from file into entries array using csv reader
entries = readEntries(csvreader, numEntries)
file.close() # Close files after use

# 3) Map & Normalize data (remove stop-words, lowercase-ify, remove punctuation & numbers & whitespace)
# 3 Method 1
# id = 0
# for entry in entries:
#     class_var = entry[0]

#     # Normalize Title
#     title_var = entry[1].lower()
#     title_var = remove_punctuation(title_var, punctuation)

#     # Normalize Description
#     description_var = entry[2].lower()
#     description_var = remove_punctuation(description_var, punctuation)

#     # Add Normalized data to map & increment ID
#     normalizedData[id] = [class_var, title_var, description_var]
#     id += 1

# 3 Method 2
for entry in entries:
    class_var = entry[0]

    # Normalize Title
    titleVar = entry[1].lower()
    titleVar = remove_punctuation(titleVar, punctuation)
    titleWords = titleVar.split(" ")

    # Normalize Description
    descriptionVar = entry[2].lower()
    descriptionVar = remove_punctuation(descriptionVar, punctuation)
    descriptionWords = descriptionVar.split(" ")

    # Get counts of words in document
    wordcount_map = {} # [word] -> [counts in document]
    for word in titleWords:
        if len(word) == 0: continue
        if word in stopwords.words(): continue
        if word_validator.check(word) == False: continue
        # if word in wordcount_map: wordcount_map[word] += 1
        # else: wordcount_map[word] = 1
        if word in termFreqOverall.keys(): termFreqOverall[word] += 1
        else: termFreqOverall[word] = 1
    
    for word in descriptionWords:
        if len(word) == 0: continue
        if word in stopwords.words(): continue
        if word_validator.check(word) == False: continue
        # if word in wordcount_map: wordcount_map[word] += 1
        # else: wordcount_map[word] = 1
        if word in termFreqOverall.keys(): termFreqOverall[word] += 1
        else: termFreqOverall[word] = 1
    
    # Update overall frequency map (handled in prior for loopss)
    # for key, value in wordcount_map.items(): # [K: word] --> [V: wordCount in document]
    #     if key in termFreqOverall.keys(): termFreqOverall[key] += value
    #     else: termFreqOverall[key] = value
    
if printStatementsOn: 
    for key, values in normalizedData.items(): print(f"{key}: {values}")

# 4) For each entry (story), map words->wordCount  ==> O(n^2)
# 4a
# for key, values in normalizedData.items():
#     wordcount_map = {} # [word] -> [counts in document]
#     id = key
#     title_words = values[1].split(" ")

#     # Method 1: Use Counter library
#     # word_frequency_dict = Counter(title_words)
#     # result_dict = dict(word_frequency_dict)

#     # Method 2: Manually
#     for word in title_words:
#         if len(word) == 0: continue
#         if word in stopwords.words():
#             if printStatementsOn: print("❌ [nltk]: ", word)
#             continue

#         if word_validator.check(word) == False: 
#             if printStatementsOn: print("❌ [enchant]: ", word)
#             continue

#         if printStatementsOn: print("✅ VALID: ", word)

#         if word in wordcount_map: wordcount_map[word] += 1
#         else: wordcount_map[word] = 1
    
#     # Update Maps
#     termFreqDict[id] = wordcount_map
#     for key, value in wordcount_map.items(): # [K: word] --> [V: wordCount in document]
#         if key in termFreqOverall.keys(): termFreqOverall[key] += value
#         else: termFreqOverall[key] = value

#     # View this term frequency map
#     if printStatementsOn: print("\n", id, " wordcount_map")
#     if printStatementsOn: 
#         for key, values in wordcount_map.items(): print(f"{key}: {values}")
#     if printStatementsOn: print("\n")

# 4b
# for key, values in normalizedData.items():
#     wordcount_map = {} # [word] -> [counts in document]
#     id = key
#     titleWords = values[1].split(" ")
#     descriptionWords = values[2].split(" ")


#     # Method 2: Manually
#     for word in titleWords:
#         if len(word) == 0: continue
#         if word in stopwords.words(): continue
#         if word_validator.check(word) == False: continue

#         if word in wordcount_map: wordcount_map[word] += 1
#         else: wordcount_map[word] = 1
    
#     # Update Maps
#     for key, value in wordcount_map.items(): # [K: word] --> [V: wordCount in document]
#         if key in termFreqOverall.keys(): termFreqOverall[key] += value
#         else: termFreqOverall[key] = value

#     # View this term frequency map
#     if printStatementsOn: print("\n", id, " wordcount_map")
#     if printStatementsOn: 
#         for key, values in wordcount_map.items(): print(f"{key}: {values}")
#     if printStatementsOn: print("\n")

termFreqOverall = dict(sorted(termFreqOverall.items(), key=lambda item: item[1], reverse=True)) # Sorting (descending) by term frequency

if printStatementsOn: # termFreqDict
    print("\ntermFreqDict\n")
    for key, values in termFreqDict.items(): print(f"{key}: {values}")
if printStatementsOn: # termFreqOverall
    print("\ntermFreqoverall\n")
    for key, values in termFreqOverall.items(): print(f"{key}: {values}")

print("\nTop 200 words vocabulary list out of", numEntries, "documents:\n")
count = 1
for key, values in termFreqOverall.items(): 
    print(f"{count}: {key}: {values}")
    if count == 200: break
    count += 1
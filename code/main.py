import sys
import matplotlib.pyplot as plt
import numpy as np
from itertools import groupby
import math
import string
import re
import os
import copy
import glob

#read_stopwords.py
def populateStopWords():
    stopWords = []
    with open('stopwords.txt') as inputFile:
        for line in inputFile:
            if(line != '\n'):
                stopWords.append(line.rstrip())
    return stopWords

#tokenizer.py
def stripWhitespace(inputString):
    return re.sub("\s+", " ", inputString.strip())

def tokenize(inputString):
    whitespaceStripped = stripWhitespace(inputString)
    punctuationRemoved = "".join([x for x in whitespaceStripped
                                  if x not in string.punctuation])
    lowercased = punctuationRemoved.lower()
    return lowercased.split()

#outputConfusionMatrix.py
def computeConfusionMatrix(predicted, groundTruth, nAuthors):
    confusionMatrix = [[0 for i in range(nAuthors+1)] for j in range(nAuthors+1)]

    for i in range(len(groundTruth)):
        confusionMatrix[predicted[i]][groundTruth[i]] += 1

    return confusionMatrix

def outputConfusionMatrix(confusionMatrix):
    columnWidth = 4

    print(str(' ').center(columnWidth),end=' ')
    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')

    print()

    for i in range(1,len(confusionMatrix)):
        print(str(i).center(columnWidth),end=' ')
        for j in range(1,len(confusionMatrix)):
            print(str(confusionMatrix[j][i]).center(columnWidth),end=' ')
        print()

#a3.py
def prior(inputFolder) :
    files = glob.glob(inputFolder + "/*train*.txt")
    temp = files[:]
    for i in range(len(temp)) :
        temp[i] = temp[i][29:31][:]
    num = [len(list(group)) for key, group in groupby(temp)]
    numFiles = 0
    for i in num :
        numFiles += i
    for i in range(len(num)) :
        num[i] = num[i]/numFiles
    return num

def conditional(inputFolder, stopWords) :
    files = glob.glob(inputFolder + "/*train*.txt")
    temp = files[:]

    for i in range(len(temp)) :
        temp[i] = temp[i][29:31][:]

    num = [len(list(group)) for key, group in groupby(temp)]

    totWords = []

    for i in range(len(files)) :
        words = []
        with open(files[i], errors='ignore') as inputFile :
            for line in inputFile :
                words.extend(tokenize(line))
        totWords.append(words[:])

    condFinal = []

    for i in range(len(num)) :
        condAuthor = []
        for j in range(len(stopWords)) :
            numStopwords = 0
            for k in range(len(temp)) :
                if((i+1) == int(temp[k])) :
                    if(stopWords[j] in totWords[k]) :
                        numStopwords += 1
            calc = numStopwords + 1
            calc /= (num[i]+2)
            condAuthor.append(calc)
        condFinal.append(condAuthor)

    return condFinal

def topFeatures(stopWords, priorVals, conditionalVals) :
    cceList = [0]*len(stopWords)
    for i in range(len(priorVals)) :
        for j in range(len(cceList)) :
            cceList[j] += (priorVals[i]*conditionalVals[i][j]*math.log(conditionalVals[i][j],2))
    for i in range(len(cceList)) :
        cceList[i] = cceList[i]*-1
    tupleList = []
    for i in range(len(cceList)) :
        tupleList.append([cceList[i],stopWords[i]])
    tupleList.sort(key=lambda x: x[1])
    tupleList.sort(key=lambda x: x[0], reverse = True)
    print("Top Features:")
    print("-------------")
    for i in range(20) :
        print(tupleList[i][1] + ": " + str(tupleList[i][0]))
    print()

def ans(inputFolder, stopWords, priorVals, conditionalVals) :
    inputFolder = sys.argv[1]
    files = glob.glob(inputFolder + "/*sample*.txt")

    predictions = []
    for i in range(len(files)) :
        featureVector = []
        words = []

        with open(files[i], errors='ignore') as inputFile :
            for line in inputFile :
                words.extend(tokenize(line))
        
        for j in range(len(stopWords)) :
            if(stopWords[j] in words) :
                featureVector.append(0)
            else :
                featureVector.append(1)

        probabilityList = []
        for j in range(len(conditionalVals)) :
            probability = 0
            for k in range(len(stopWords)) :
                if(featureVector[k] == 0) :
                    probability += math.log(conditionalVals[j][k],2)
                else :
                    probability += math.log(1-conditionalVals[j][k],2)
            probability += math.log(priorVals[j],2)
            probabilityList.append(probability)

        predictions.append(probabilityList.index(max(probabilityList)) + 1)
    return predictions

def ground(inputFolder) :
    parse = []
    actual = []
    with open("test_ground_truth.txt") as inputFile :
        for line in inputFile :
            if(line[0:8] == inputFolder[14:22]) :
                parse.append(line[:])
    
    for i in parse :
        temp = int(i[29:])
        actual.append(temp)
    return actual

def calcAcc(predictions, actual) :
    numCorrect = 0
    for i in range(len(predictions)) :
        if(predictions[i] == actual[i]) :
            numCorrect += 1
    print("Accuracy:")
    print("---------")
    print(numCorrect/len(predictions))
    print()

def calcAcc2(predictions, actual) :
    numCorrect = 0
    for i in range(len(predictions)) :
        if(predictions[i] == actual[i]) :
            numCorrect += 1
    print(numCorrect/len(predictions))
    return numCorrect/len(predictions)

def printConfuse(predictions, actual) :
    print("Confusion Matrix:")
    print("-----------------")
    outputConfusionMatrix(computeConfusionMatrix(predictions, actual, len(actual)))
    print()

def ffHelper(inputFolder, stopWords) :
    priorVals = prior(inputFolder)
    conditionalVals = conditional(inputFolder, stopWords)
    predictions = ans(inputFolder, stopWords, priorVals, conditionalVals)
    actual = ground(inputFolder)
    acc = calcAcc2(predictions, actual)
    return acc

def graphFeatures(inputFolder, points) :
    x = []
    y = []
    for i in points :
        x.append(i[0])
        y.append(i[1])
    
    plt.plot(x, y, '-r')
    plt.savefig(inputFolder)

def frequentFeatures(inputFolder, stopWords) :
    tuples = []
    points = []
    inputFolder = sys.argv[1]
    files = glob.glob(inputFolder + "/*train*.txt")

    words = []

    for i in range(len(files)) :
        with open(files[i], errors='ignore') as inputFile :
            for line in inputFile :
                words.extend(tokenize(line))
    
    for i in range(len(stopWords)) :
        tuples.append([words.count(stopWords[i]),stopWords[i]])
    
    tuples.sort(key=lambda x: x[1])
    tuples.sort(key=lambda x: x[0], reverse = True)

    print("Training w/ Frequent Features")
    print("-----------------------------")
    for i in range(10,len(tuples),10) :
        print(str(i) + ": ", end = '')
        tempWords = []
        for j in range(i) :
            tempWords.append(tuples[j][1])
        acc = ffHelper(inputFolder, tempWords)
        points.append([i,acc])
    
    graphFeatures(inputFolder, points)
    
def main() :
    #get all the stopwords into a list
    stopWords = populateStopWords()
    #remove duplicates
    stopWords = list(dict.fromkeys(stopWords))
    #get folder
    inputFolder = sys.argv[1]
    #compute prior
    priorVals = prior(inputFolder)
    #compute conditional per feature
    conditionalVals = conditional(inputFolder, stopWords)
    #determine the c values for each sample
    predictions = ans(inputFolder, stopWords, priorVals, conditionalVals)
    #parse groundTruths
    actual = ground(inputFolder)
    #calculate accuracy
    calcAcc(predictions, actual)
    #generate confusion matrix
    printConfuse(predictions, actual)
    #determine top features
    topFeatures(stopWords, priorVals, conditionalVals)
    #generate feature graphs
    frequentFeatures(inputFolder, stopWords)

main()
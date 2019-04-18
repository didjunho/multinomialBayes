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
def bigrams() :
    files = glob.glob(sys.argv[1] + "/train2000-*")
    bigramList = []
    lineList = []
    for i in files :
        f = open(i, encoding = "iso-8859-15", errors = 'ignore')
        for j in f :
            j = j.rstrip('\n')
            lineList.append(j.lower())
    for i in lineList :
        for j in range(len(i)-1) :
            bigramList.append(str(i[j]) + str(i[j+1]))
    return bigramList, files

def trigrams() :
    files = glob.glob(sys.argv[1] + "/train2000-*")
    trigramList = []
    lineList = []
    for i in files :
        f = open(i, encoding = "iso-8859-15", errors = 'ignore')
        for j in f :
            j = j.rstrip('\n')
            lineList.append(j.lower())
    for i in lineList :
        for j in range(len(i)-2) :
            trigramList.append(str(i[j]) + str(i[j+1]) + str(i[j+2]))
    return trigramList

def conditional(stopWords, f) :
    condFinal = []

    for i in f :
        tempCond = []
        files = open(i, encoding = "iso-8859-15", errors = 'ignore')
        lines = []
        allTimes = 0
        for j in files :
            j = j.rstrip('\n')
            lines.append(j.lower())
        for j in stopWords :
            for k in lines :
                allTimes += k.count(j)
        for j in stopWords :
            times = 0
            for k in lines :
                times += k.count(j)
            fin = (times+1)/(allTimes + len(stopWords))
            tempCond.append(fin)
        condFinal.append(tempCond)
    
    return condFinal

def ans(stopWords, conditionalVals) :
    mand = math.log(.25,2)
    files = glob.glob(sys.argv[1] + "/test-*")
    predictions = []
    for i in range(len(files)) :
        tempList = []
        featureVector = []
        f = open(files[i], encoding = "utf-8", errors = 'ignore')
        for j in f :
            j.rstrip('\n')
            tempList.append(j.lower())
        for j in stopWords :
            found = False
            for k in tempList :
                if(j in k) :
                    featureVector.append(1)
                    found = True
                    break
            if(found == False) :
                featureVector.append(0)
        probabilityList = []
        for j in range(len(conditionalVals)) :
            probability = 0
            for k in range(len(stopWords)) :
                if(featureVector[k] == 1) :
                    probability += math.log(conditionalVals[j][k],2)
                else :
                    probability += math.log(1-conditionalVals[j][k],2)
            probability += mand
            probabilityList.append(probability)
        predictions.append(probabilityList.index(max(probabilityList)) + 1)
    return predictions

def calcAcc(predictions, actual) :
    numCorrect = 0
    for i in range(len(predictions)) :
        if(predictions[i] == actual[i]) :
            numCorrect += 1
    print("Accuracy:")
    print("---------")
    print(numCorrect/len(predictions))
    print()

def printConfuse(predictions, actual) :
    print("Confusion Matrix:")
    print("-----------------")
    outputConfusionMatrix(computeConfusionMatrix(predictions, actual, len(actual)))
    print()

def main() :
    print("program takes approximately 60 seconds to run on mid-spec laptop")
    #get bigram list
    bstopWords, f = bigrams()
    #get trigram list
    tstopWords = trigrams()
    #remove duplicates
    bstopWords = list(dict.fromkeys(bstopWords))
    tstopWords = list(dict.fromkeys(tstopWords))

    #compute conditional per feature
    bconditionalVals = conditional(bstopWords, f)
    tconditionalVals = conditional(tstopWords, f)

    #determine the c values for each sample
    bpredictions = ans(bstopWords, bconditionalVals)
    tpredictions = ans(tstopWords, tconditionalVals)
    #parse groundTruths
    actual = [1,2,3,4]
    #print acc/cofusion matrix

    print("bigrams:")
    print()
    calcAcc(bpredictions, actual)
    printConfuse(bpredictions, actual)

    print("trigrams:")
    print()
    calcAcc(tpredictions, actual)
    printConfuse(tpredictions, actual)

main()
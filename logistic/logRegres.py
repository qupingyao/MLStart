#-*- coding: UTF-8 -*-
from math import *
from numpy import *
import matplotlib.pyplot as plt

def loadDataSet():
    dataMat = []
    labelMat = []
    fr = open('testSet.txt')
    for line in fr.readlines():
        lineArr = line.strip().split()
        dataMat.append([1.0, float(lineArr[0]), float(lineArr[1])])
        labelMat.append(int(lineArr[2]))
    return dataMat,labelMat

def sigmoid(inX):
    return 1.0/(1 + exp(-inX))

def gradAscent(dataMatIn,classLabels):
    dataMatrix = mat(dataMatIn)
    labelMat = mat(classLabels).transpose()
    m,n = shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = ones((n,1))
    for k in range(maxCycles):
        h = sigmoid(dataMatrix * weights)
        error = (labelMat - h)
        weights = weights + alpha * dataMatrix.transpose()*error
    return weights

def plotBestFit(weights):
    dataMat,labelMat = loadDataSet()
    dataArr = array(dataMat)
    n = shape(dataArr)[0]
    xcrod1 = [];ycord1 = []
    xcrod2 = [];ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcrod1.append(dataArr[i,1])
            ycord1.append(dataArr[i,2])
        else:
            xcrod2.append(dataArr[i,1])
            ycord2.append(dataArr[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcrod1, ycord1, s=30, c='red', marker='s')
    ax.scatter(xcrod2, ycord2, s=30, c='green')
    x = arange(-3.0,3.0,0.1)
    weightsT = array(weights)
    y = (-weightsT[0] - weightsT[1] * x) / weightsT[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    alpha = 0.01
    weights = ones(n)
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + alpha * error * dataMatrix[i]
    return weights

def stocGradAscent1(dataMatrix,classLabels,numIter = 150):
    m,n = shape(dataMatrix)
    weights = ones(n)
    for j in range(numIter):
        dataIndex = range(m)
        for i in range(m):
            alpha = 4/(1.0 + j + i)+0.01
            randIndex = int(random.uniform(0,len(dataIndex)))
            h = sigmoid(sum(dataMatrix[randIndex]*weights))
            error = classLabels[randIndex] - h
            weights = weights + alpha * error * dataMatrix[randIndex]
            del(dataIndex[randIndex])
    return weights

#预测病马死亡实例
def classifyVector(inX,weights):
    prob = sigmoid(sum(inX*weights))
    if prob > 0.5:
        return 1.0
    else:
        return 0.0

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(array(trainingSet),trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def oldColicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscent(array(trainingSet),trainingLabels)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def newColicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainAverageSet = [0]*21
    trainSumSet = [0]*21
    trainCountSet = [0]*21
    testAverageSet = [0] * 21
    testSumSet = [0] * 21
    testCountSet = [0] * 21
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        for i in range(21):
            if float(currLine[i])!=0.0:
                trainCountSet[i] += 1.0
                trainSumSet[i] += float(currLine[i])
    for i in range(21):
        trainAverageSet[i] = trainSumSet[i]/trainCountSet[i]
    frTrain.close()
    frTrain = open('horseColicTraining.txt')
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            if float(currLine[i])!=0.0:
                lineArr.append(float(currLine[i]))
            else:
                lineArr.append(float(trainAverageSet[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = gradAscent(array(trainingSet),trainingLabels)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        for i in range(21):
            if float(currLine[i])!=0.0:
                testCountSet[i] += 1.0
                testSumSet[i] += float(currLine[i])
    for i in range(21):
        testAverageSet[i] = testSumSet[i]/testCountSet[i]
    frTest.close()
    frTest = open('horseColicTest.txt')
    for line in frTest.readlines():
        numTestVec += 1.0
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            if float(currLine[i])!=0.0:
                lineArr.append(float(currLine[i]))
            else:
                lineArr.append(float(testAverageSet[i]))
        if int(classifyVector(array(lineArr),trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = (float(errorCount)/numTestVec)
    print "the error rate of this test is: %f" % errorRate
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print "after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests))

def oldMultiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += oldColicTest()
    print "after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests))

def newMultiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += newColicTest()
    print "after %d iterations the average error rate is: %f" % (numTests,errorSum/float(numTests))

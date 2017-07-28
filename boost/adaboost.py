#-*- coding: UTF-8 -*-
from numpy import *
from matplotlib import pyplot as plt

def loadSimpData():
    datMat = matrix([[1,2.1],[2,1.1],[1.3,1],[1,1],[2,1]])
    classLabels = [1.0,1.0,-1.0,-1.0,1.0]
    return datMat,classLabels

def stumpClassify(dataMatrix,dimen,threshVal,threshIneq):
    retArray = ones((shape(dataMatrix)[0],1))
    if threshIneq == 'lt':
        retArray[dataMatrix[:,dimen] <= threshVal] = -1.0
    else:
        retArray[dataMatrix[:,dimen] > threshVal] = -1.0
    return retArray

def buildStump(dataArr,classLabels,D):
    dataMatrix = mat(dataArr)
    labelMat = mat(classLabels).T
    m,n = shape(dataMatrix)
    numSteps = 10.0
    bestStump = {}
    bestClasEst = mat(zeros((m,1)))
    minError = inf
    for i in range(n):
        rangeMin = dataMatrix[:,i].min()
        rangeMax = dataMatrix[:,i].max()
        stepSize = (rangeMax - rangeMin)/numSteps
        for j in range(-1,int(numSteps)+1):
            for inequal in ['lt','gt']:
                threshVal = (rangeMin + float(j) * stepSize)
                predictedVals = stumpClassify(dataMatrix,i,threshVal,inequal)
                errArr = mat(ones((m,1)))
                errArr[predictedVals == labelMat] = 0
                weightedError = D.T*errArr
                print "split: dim %d, thresh %.2f, thresh inequal: %s, the weighted error is %.3f" %(i,threshVal,inequal,weightedError)
                if weightedError < minError:
                    minError = weightedError
                    bestClasEst = predictedVals
                    bestStump['dim'] = i
                    bestStump['thresh'] = threshVal
                    bestStump['ineq'] = inequal
    return bestStump,minError,bestClasEst

def adaBoostTrainDS(dataArr,classLabels,numIt=40):
    weakClassArr = []
    m = shape(dataArr)[0]
    D = mat(ones((m,1))/m)
    aggClassEst = mat(zeros((m,1)))
    for i in range(numIt):
        bestStump,error,classEst = buildStump(dataArr,classLabels,D)
        print "D:",D.T
        alpha = float(0.5*log((1.0 - error) / max(error,1e-16)))
        bestStump['alpha'] = alpha
        weakClassArr.append(bestStump)
        print "classEst: ",classEst.T
        expon = multiply(-1*alpha*mat(classLabels).T,classEst)
        D = multiply(D,exp(expon))
        D = D/D.sum()
        aggClassEst += alpha*classEst
        print "aggClassEst: ",aggClassEst.T
        aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T,ones((m,1)))
        errorRate = aggErrors.sum()/m
        print "total error: ",errorRate,"\n"
        if errorRate == 0.0:
            break
    return weakClassArr,aggClassEst

def adaClassify(datToclass,classifierArr):
    dataMatrix = mat(datToclass)
    m = shape(dataMatrix)[0]
    aggClassEst = mat(zeros((m,1)))
    for i in range(len(classifierArr)):
        classEst = stumpClassify(dataMatrix,classifierArr[i]['dim'],classifierArr[i]['thresh'],classifierArr[i]['ineq'])
        aggClassEst += classifierArr[i]['alpha']*classEst
        print aggClassEst
    return sign(aggClassEst)

#预测病马死亡实例
def loadDataSet(fileName):
    a=open(fileName).readline()
    numFeat = len(open(fileName).readline().split('\t'))
    dataMat = []
    labelMat = []
    fr = open(fileName)
    for line in fr.readlines():
        lineArr = []
        curLine = line.strip().split('\t')
        for i in range(numFeat-1):
            lineArr.append(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

def newColicTest():
    frTrain = open('horseColicTraining2.txt')
    frTest = open('horseColicTest2.txt')
    trainAverageSet = [0]*21
    trainSumSet = [0]*21
    trainCountSet = [0]*21
    testAverageSet = [0] * 21
    testSumSet = [0] * 21
    testCountSet = [0] * 21
    trainingSet = []
    trainingLabels = []
    testingSet = []
    testingLabels = []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        for i in range(21):
            if float(currLine[i])!=0.0:
                trainCountSet[i] += 1.0
                trainSumSet[i] += float(currLine[i])
    for i in range(21):
        trainAverageSet[i] = trainSumSet[i]/trainCountSet[i]
    frTrain.close()
    frTrain = open('horseColicTraining2.txt')
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
    classifierArray,ageClassEst = adaBoostTrainDS(trainingSet, trainingLabels, 10)
    numTestVec = 0
    for line in frTest.readlines():
        currLine = line.strip().split('\t')
        for i in range(21):
            if float(currLine[i])!=0.0:
                testCountSet[i] += 1.0
                testSumSet[i] += float(currLine[i])
    for i in range(21):
        testAverageSet[i] = testSumSet[i]/testCountSet[i]
    frTest.close()
    frTest = open('horseColicTest2.txt')
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            if float(currLine[i])!=0.0:
                lineArr.append(float(currLine[i]))
            else:
                lineArr.append(float(testAverageSet[i]))
        testingSet.append(lineArr)
        testingLabels.append(float(currLine[21]))
    prediction10 = adaClassify(testingSet, classifierArray)
    errArr = mat(ones((numTestVec, 1)))
    return errArr[prediction10 != mat(testingLabels).T].sum() / numTestVec

#预测ROC曲线实例
def plotROC(predStrengths, classLabels):
    cur = (1.0,1.0)
    ySum = 0.0
    numPosClas = sum(array(classLabels) == 1.0)
    yStep = 1/float(numPosClas)
    xStep = 1/float(len(classLabels) - numPosClas)
    sortedIndicies = predStrengths.argsort()
    fig = plt.figure()
    fig.clf()
    ax = plt.subplot(111)
    for index in sortedIndicies.tolist()[0]:
        if classLabels[index] == 1.0:
            delX = 0
            delY = yStep
        else:
            delX = xStep
            delY = 0
            ySum += cur[1]
        ax.plot([cur[0],cur[0]-delX],[cur[1],cur[1]-delY], 'b')
        cur = (cur[0] - delX,cur[1]-delY)
    ax.plot([0,1],[0,1],'b--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('Ture Positive Rate')
    plt.title('ROC curve for AdaBoost Horse Colic Detection System')
    ax.axis([0,1,0,1])
    plt.show()
    print "the Area Under the Curve is: ",ySum * xStep

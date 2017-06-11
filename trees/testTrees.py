#-*- coding: UTF-8 -*-
from trees import *
from treePlotter import *

# myDat,labels = createDataSet()
# tempLabels = labels[:]
# print createTree(myDat,labels)
# createPlot()
# myTree = retrieveTree(0)
# createNewPlot(myTree)
# print classify(myTree,tempLabels,[1,0])
# print classify(myTree,tempLabels,[1,1])
# storeTree(myTree,'classifierStorage.txt')
# print grabTree('classifierStorage.txt')

#使用决策树预测隐形眼镜实例
fr = open('lenses.txt')
lenses = [inst.strip().split('\t') for inst in fr.readlines()]
lensesLabels = ['age','prescript','astigmatic','tearRate']
lensesTree = createTree(lenses,lensesLabels)
print lensesTree
createNewPlot(lensesTree)



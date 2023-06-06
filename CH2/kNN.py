from numpy import *
import operator

def createDataSet():
    group = array([[1.0,1.1],
                   [1.0,1.0],
                   [0,0],
                   [0,0.1]])
    labels = ['A',
              'A',
              'B',
              'B']
    return group, labels

def classify0(inX, dataSet, labels, k):
    """classify the input data inX to catergory different in labels

    Args:
        inX (array): test data
        dataSet (array): train sets
        labels (array): labels of train sets
        k (int): k point decide the catergory of inX

    Returns:
        int: class of inX
    """
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount={}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    sortedClassCount = sorted(classCount.items(),
                              key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def file2matrix(filename):
    """read datasets from file to matrix

    Args:
        filename (string): data file

    Returns:
        returnMat: datasets in the datafile, array
        classLabelVector: label of returnMat, one arow array
    """
    love_dictionary = {'largeDoses':3, 'smallDoses':2, 'didntLike':1}
    fr = open(filename)
    arrayOLines = fr.readlines()
    numberOfLines = len(arrayOLines)
    returnMat = zeros((numberOfLines,3))
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        listFromLine = line.split('\t')
        returnMat[index,:] = listFromLine[0:3]
        if(listFromLine[-1].isdigit()):
            classLabelVector.append(int(listFromLine[-1]))
        else:
            classLabelVector.append(love_dictionary.get(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector

def autoNorm(dataSet) :
    """normalization datasets

    Args:
        dataSet (array): datasets with different feature

    Returns:
        normDataSet: array, normalized datasets
        ranges: vector, ranges of each feature
        minVals: vector, min data of each feature
    """
    minVals = dataSet.min(0)
    maxVals = dataSet.max(0)
    ranges = maxVals - minVals
    normDataSet = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normDataSet = dataSet - tile(minVals, (m, 1))
    normDataSet = normDataSet/tile(ranges, (m, 1))
    return normDataSet, ranges, minVals

def datingClassTest():
    hoRatio = 0.1 # ratio of test set
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt') # extract data and label
    normMat, ranges, minVals = autoNorm(datingDataMat) # normalize data
    m = normMat.shape[0] # get the sample point num
    numTestVecs = int(m*hoRatio) # get the index of test set
    errorCount = 0.0 # counting the classification error sample num
    for i in range (numTestVecs):
        classifierResult = classify0(normMat[i,:], normMat[numTestVecs:m, :],\
                                     datingLabels[numTestVecs:m], 3)
        print("the classifier came back with: %d, the real answer is: %d"\
              % (classifierResult, datingLabels[i]))
        if (classifierResult != datingLabels[i]): errorCount += 1.0 # counting error classified point
        print("the total error rate is: %f" % (errorCount/float(numTestVecs)))

def classifyPerson():
    resultList = ['not at all', 'in small doses', 'in large doses']
    percentTats = float(input("percentage of time spent playing video games?"))
    ffMiles = float(input("frequent flier miles earned per year?"))
    iceCream = float(input("liters of ice cream consumed per year?"))
    datingDataMat, datingLabels = file2matrix('datingTestSet.txt')
    normMat, ranges, minVals = autoNorm(datingDataMat)
    inArr = array([ffMiles, percentTats, iceCream])
    classifierResult = classify0((inArr-minVals)/ranges, normMat, datingLabels, 3)
    print("You will probably like this person: ", resultList[classifierResult-1])

def img2vector(filename):
    returnVect = zeros((1,1024))
    fr = open(filename)
    for i in range(32):
        lineStr = fr.readline()
        for j in range(32):
            returnVect[0,32*i+j] = int(lineStr[j])
    return returnVect

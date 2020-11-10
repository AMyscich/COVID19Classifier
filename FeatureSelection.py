import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.feature_selection import RFE, mutual_info_classif, SelectKBest
import time
import matplotlib.pyplot as plt

class FeatureSelection:
    def __init__(self):
        return

    # Wrapper method of feature selection using RFE
    def wrapperMethod(self, data, labels, numFeatures=100):
        model = LogisticRegression(solver='liblinear', max_iter=100)
        selector = RFE(model, n_features_to_select=numFeatures, step=0.1)
        selector = selector.fit(data, labels)

        return (selector.transform(data), labels, selector.ranking_)

    # Filter method of feature selection using mutual_info_classif
    def filterMethod(self, data, labels, numFeatures=100):
        selector = SelectKBest(mutual_info_classif, k=numFeatures).fit(data, labels)

        return (selector.transform(data), labels)

    # Gets best features for wrapper and filter methods
    # Returns best features from wrapper and filter method
    def getBestFeatures(self, trainingData, trainingLabels, epochs=100, folds=5, numFeatures=2000):
        trainingData = trainingData.reshape((trainingData.shape[0], np.prod(np.array(list(trainingData.shape)[1:]))))

        startTime = time.time()
        wrapperFeatures, wrapperLabels, rankings = self.wrapperMethod(trainingData, trainingLabels, numFeatures)
        print("Wrapper method run time: ", time.time() - startTime)
        startTime = time.time()
        filterFeatures, filterLabels = self.filterMethod(trainingData, trainingLabels, numFeatures)
        print("Filter method run time: ", time.time() - startTime)

        model = LogisticRegression(solver='liblinear', max_iter=epochs, random_state=234)
        wrapperCVAcc = cross_val_score(model, wrapperFeatures, wrapperLabels, cv=folds)
        filterCVAcc = cross_val_score(model, filterFeatures, filterLabels, cv=folds)

        print("Accuracy for wrapper method: ", np.average(wrapperCVAcc), "Accuracy for filter method: ", np.average(filterCVAcc))
        return (wrapperFeatures, filterFeatures)

    # Creates graphs comparing number of features to performance for both filter and wrapper methods
    # featureInterval is the number of features to add between models
    def getGraphs(self, trainingData, trainingLabels, epochs=100, folds=5, featureInterval=100):
        trainingData = trainingData.reshape((trainingData.shape[0], np.prod(np.array(list(trainingData.shape)[1:]))))

        wrapperPerformance = []
        filterPerformance = []
        labels = []

        filterRanking = mutual_info_classif(trainingData, trainingLabels)
        wrapperRanking = self.wrapperMethod(trainingData, trainingLabels, 1)[2]

        filterIndices = sorted(range(len(filterRanking)), key=lambda x: filterRanking[x], reverse=True)
        wrapperIndices = sorted(range(len(wrapperRanking)), key=lambda x: wrapperRanking[x])

        for num in range(1, trainingData.shape[1], featureInterval):
            newFilterData = []
            newWrapperData = []

            for sample in trainingData:
                newFilterData.append([sample[x] for x in filterIndices[:num]])
                newWrapperData.append([sample[x] for x in wrapperIndices[:num]])

            model = LogisticRegression(solver='liblinear', max_iter=epochs, random_state=234)
            wrapperCVAcc = cross_val_score(model, np.array(newWrapperData), trainingLabels, cv=folds)
            filterCVAcc = cross_val_score(model, np.array(newFilterData), trainingLabels, cv=folds)

            #print("Accuracy for wrapper method: ", np.average(wrapperCVAcc), "Accuracy for filter method: ", np.average(filterCVAcc))
            
            wrapperPerformance.append(np.average(wrapperCVAcc))
            filterPerformance.append(np.average(filterCVAcc))
            labels.append(num)

        self.saveGraph("Wrapper Method Performance", labels, wrapperPerformance, "WrapperMethod.png")
        self.saveGraph("Filter Method Performance", labels, filterPerformance, "FilterMethod.png")

    # Creates and saves a graph given the data and labels
    def saveGraph(self, title, xData, yData, filename):
        plt.scatter(xData, yData)
        plt.plot(xData, yData)
        plt.title(title)
        plt.xlabel("Number of Features")
        plt.ylabel("Accuracy")
        plt.savefig(filename)
        plt.close()

    # Overlays original image with mask that highlights the pixels selected by feature selection
    def highlightImage(self, data, labels, numFeatures, origSize):
        reshapedData = data.reshape((data.shape[0], np.prod(np.array(list(data.shape)[1:]))))

        newData, labels, rankings = self.wrapperMethod(reshapedData, labels, numFeatures)      

        newImage = np.array(rankings).reshape(origSize, origSize)
        mask = np.ma.masked_where(newImage != 1, newImage)

        for image in data:
            plt.imshow(image.reshape(origSize, origSize), cmap='gray', interpolation='none')
            plt.imshow(mask, cmap='jet', alpha=0.5, interpolation='none')
            plt.show()
            plt.close()

## Example Usage
#import pandas
#import cv2
#from ImageRaster import ImageRaster
#from FeatureExtract import FeatureExtract

#train = pandas.read_csv('train.csv', header=0)
#trainData = []
#trainLabels = []

#fe = FeatureExtract()
#fs = FeatureSelection()

#for index, row in train.iterrows():
#    filename = 'train/' + row['filename']
#    newImage = ImageRaster.read_resized_image(filename, 128)
#    newImage = cv2.cvtColor(newImage, cv2.COLOR_RGB2GRAY)
#    #temp = []
#    #g, gx, gy = fe.edge_detection(newImage)
#    #g_smooth = fe.smooth_image(g)
#    #temp.append(g)
#    #temp.append(gx)
#    #temp.append(gy)
#    #temp.append(g_smooth)
#    #temp = np.array(temp)
#    trainData.append(newImage)

#    trainLabels.append(row[-1])

#trainData = np.array(trainData)
#trainLabels = np.array(trainLabels)

## Use this to get the best features for wrapper and filter methods, returns tuple (wrapperfeatures, filterfeatures)
#wrapperFeatures, filterFeatures = fs.getBestFeatures(trainData, trainLabels)

##Use this to get performance graphs
##fs.getGraphs(trainData, trainLabels, 100, 5, 1000)

## trainData must be dimension 2 for this function
##fs.highlightImage(trainData, trainLabels, 2000, trainData.shape[1])
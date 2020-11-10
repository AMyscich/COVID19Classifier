#from COVID19Classifier.ImageRaster import ImageRaster
from DataVisualization import DataVisualization
from DataHandler import DataHandler
from ImageRaster import ImageRaster
from FeatureExtract import FeatureExtract
from FeatureSelection import FeatureSelection
from EnsembleLearning import EnsembleLearning
import pandas as pd
import os
import cv2
import numpy as np



def main():
    dv = DataVisualization()
    dh = DataHandler()
    el = EnsembleLearning()

    dataframe_train = pd.read_csv("data" + os.path.sep + "train.csv")
    dataframe_test = pd.read_csv("data" + os.path.sep + "test.csv")

    # pd.set_option('max_columns', None)
    # print(dataframe_train)

    train = pd.read_csv('data/train.csv', header=0)

    trainData = []
    trainLabels = []

    fe = FeatureExtract()
    fs = FeatureSelection()

    for index, row in train.iterrows():
       filename = 'data/train/' + row['filename']
       newImage = ImageRaster.read_resized_image(filename, 128)
       newImage = cv2.cvtColor(newImage, cv2.COLOR_RGB2GRAY)
       data = []
       g, gx, gy = fe.edge_detection(newImage)
       g_smooth = fe.smooth_image(g)
       data.append(g)
       data.append(gx)
       data.append(gy)
       data.append(g_smooth)
       data = np.array(data)
       trainData.append(data)
       trainLabels.append(row[-1])

    trainData = np.array(trainData)
    trainLabels = np.array(trainLabels)
    dh.div("Image pre-processing, a")

    dh.div("Visual feature extraction, a")

    dh.div("Feature exploration, b.i")

    dh.div("Image pre-processing, b.ii")
    dv.dataSummary(dataframe_train, drop_series=["filename"])
    dv.boxPlot(dataframe_train, "age", "covid(label)")
    dv.boxPlot(dataframe_train, "age", "gender")
    dv.swarmPlot(dataframe_train, "covid(label)", "age", "gender")

    dh.div("Feature selection, b.iii")

    dh.div("Ensemble learning, b.iV")
    el.runAdaBoost(trainData, trainLabels)
    # el.grid_search(trainData, trainLabels)
    dh.div("Improving performance, c")

    dh.div("Best Models, d")




if __name__ == "__main__":
    main()
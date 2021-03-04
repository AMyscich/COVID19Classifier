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
import matplotlib.pyplot as plt


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
       newImage = ImageRaster.read_resized_image(filename, 256)
       newImage = cv2.cvtColor(newImage, cv2.COLOR_RGB2GRAY)
       data = []
       image_smooth = fe.smooth_image(newImage, sd=2.0)
       image_eq = fe.equalize_image(image_smooth)
       image_hog, vis = fe.get_hog(image_eq)

       highlighted = fe.equalize_image(vis)
       plt.imshow(highlighted, cmap='gray', vmin=0, vmax=255)
       plt.title('HoG')
       plt.show()

       data.append(image_hog)
       # g, gx, gy = fe.edge_detection(newImage)
       # g_smooth = fe.smooth_image(g)
       # data.append(g)
       # data.append(gx)
       # data.append(gy)
       # data.append(g_smooth)
       data = np.array(data)
       trainData.append(data)
       trainLabels.append(row[-1])

    trainData = np.array(trainData)
    trainLabels = np.array(trainLabels)

    neg_data = np.squeeze(trainData[np.argwhere(trainLabels == 0)])
    pos_data = np.squeeze(trainData[np.argwhere(trainLabels == 1)])

    random_indices = np.random.permutation(neg_data.shape[0])
    neg_data = neg_data[random_indices]
    random_indices = np.random.permutation(pos_data.shape[0])
    pos_data = pos_data[random_indices]

    n_folds = 5
    cl = 3
    n_pos_data = pos_data.shape[0]
    n_neg_data = neg_data.shape[0]

    if n_pos_data > n_neg_data:
        n_pos_data = n_neg_data
    else:
        n_neg_data = n_pos_data

    pos_data = pos_data[:n_pos_data]
    neg_data = neg_data[:n_neg_data]


    pos_per_fold = int(n_pos_data//n_folds)
    neg_per_fold = int(n_neg_data//n_folds)


    accs = []
    for fold in range(n_folds):
        pos_data_test = pos_data[fold*pos_per_fold:(fold+1)*pos_per_fold]
        pos_data_train = np.concatenate((pos_data[:fold*pos_per_fold], pos_data[(fold+1)*pos_per_fold:]))
        neg_data_test = neg_data[fold*neg_per_fold:(fold+1)*neg_per_fold]
        neg_data_train = np.concatenate((neg_data[:fold*neg_per_fold], neg_data[(fold+1)*neg_per_fold:]))

        print(pos_data_train.shape[0])
        print(neg_data_train.shape[0])

        neg_filter = fe.generate_matched_filter(neg_data_train, cl=cl)
        pos_filter = fe.generate_matched_filter(pos_data_train, cl=cl)

        plt.imshow(pos_filter, cmap='gray', vmin=0, vmax=255)
        plt.show()
        plt.imshow(neg_filter, cmap='gray', vmin=0, vmax=255)
        plt.show()

        n_correct = 0
        for datum in range(pos_data_test.shape[0]):

            p, _, _ = fe.apply_matched_filter(neg_filter, pos_filter, pos_data_test[datum])
            if p == 1:
                n_correct += 1
        pos_acc = n_correct/pos_data_test.shape[0]
        print('Positive acc: ', pos_acc)


        n_correct = 0
        for datum in range(neg_data_test.shape[0]):

            p, _, _ = fe.apply_matched_filter(neg_filter, pos_filter, neg_data_test[datum])
            if p == 0:
                n_correct += 1

        neg_acc = n_correct/neg_data_test.shape[0]
        print('Negative acc: ', neg_acc)

        accs.append((pos_acc+neg_acc)/2)

    print(accs)


    # dh.div("Image pre-processing, a")
    #
    # dh.div("Visual feature extraction, a")
    #
    # dh.div("Feature exploration, b.i")
    #
    # dh.div("Image pre-processing, b.ii")
    # dv.dataSummary(dataframe_train, drop_series=["filename"])
    # dv.boxPlot(dataframe_train, "age", "covid(label)")
    # dv.boxPlot(dataframe_train, "age", "gender")
    # dv.swarmPlot(dataframe_train, "covid(label)", "age", "gender")
    #
    # dh.div("Feature selection, b.iii")
    #
    # dh.div("Ensemble learning, b.iV")
    # el.runAdaBoost(trainData, trainLabels)
    # # el.grid_search(trainData, trainLabels)
    # dh.div("Improving performance, c")
    #
    # dh.div("Best Models, d")




if __name__ == "__main__":
    main()

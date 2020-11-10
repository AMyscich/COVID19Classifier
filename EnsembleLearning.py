import numpy as np
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import time
import matplotlib.pyplot as plt


class EnsembleLearning:
    def __init__(self):
        return
    
    # AdaBoost with different classifier.
    def runAdaBoost(self, trainData, trainLabels, n_estimators=50, learning_rate=1, epochs=100, folds=5):
      trainData = trainData.reshape((trainData.shape[0], np.prod(np.array(list(trainData.shape)[1:]))))

      model = AdaBoostClassifier(n_estimators = n_estimators, learning_rate = learning_rate)
      scores = cross_val_score(model, trainData, trainLabels, scoring='accuracy', cv = folds)
      print("Validation accuracy for AdaBoost method with decision tree classifier: ", np.average(scores))

      svc=SVC(probability=True, kernel='linear')
      model = AdaBoostClassifier(base_estimator=svc, n_estimators = n_estimators, learning_rate = learning_rate)
      scores = cross_val_score(model, trainData, trainLabels, scoring='accuracy', cv = folds)
      print("Validation accuracy for AdaBoost method with support vector classifier: ", np.average(scores))
    
    # Grid search on Adaboost with support vector classifier
    def grid_search(self, trainData, trainLabels, folds=5):
      trainData = trainData.reshape((trainData.shape[0], np.prod(np.array(list(trainData.shape)[1:]))))
      svc=SVC(probability=True, kernel='linear')
      model = AdaBoostClassifier(base_estimator=svc)

      grid = dict()
      grid['n_estimators'] = [10, 50, 100, 500]
      grid['learning_rate'] = [0.0001, 0.001, 0.01, 0.1, 1.0]

      grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=folds, scoring='accuracy')
      grid_result = grid_search.fit(trainData, trainLabels)
      print("AdaBoost method with support vector classifier Best performance: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

# # Example Usage
# import pandas
# import cv2
# from ImageRaster import ImageRaster
# from FeatureExtract import FeatureExtract
# from FeatureSelection import FeatureSelection

# train = pandas.read_csv('data/train.csv', header=0)

# trainData = []
# trainLabels = []

# fe = FeatureExtract()
# fs = FeatureSelection()

# for index, row in train.iterrows():
#    filename = 'data/train/' + row['filename']
#    newImage = ImageRaster.read_resized_image(filename, 128)
#    newImage = cv2.cvtColor(newImage, cv2.COLOR_RGB2GRAY)
#    data = []
#    g, gx, gy = fe.edge_detection(newImage)
#    g_smooth = fe.smooth_image(g)
#    data.append(g)
#    data.append(gx)
#    data.append(gy)
#    data.append(g_smooth)
#    data = np.array(data)
#    trainData.append(data)
#    trainLabels.append(row[-1])

# trainData = np.array(trainData)
# trainLabels = np.array(trainLabels)
# el = EnsembleLearning()
# el.runAdaBoost(trainData, trainLabels)
# # el.grid_search(trainData, trainLabels)
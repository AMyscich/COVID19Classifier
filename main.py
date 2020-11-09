#from COVID19Classifier.ImageRaster import ImageRaster
from COVID19Classifier.DataVisualization import DataVisualization
from COVID19Classifier.DataHandler import DataHandler
import pandas as pd
import os


def main():
    dv = DataVisualization()
    dh = DataHandler()
    dataframe_train = pd.read_csv("data" + os.path.sep + "train.csv")
    dataframe_test = pd.read_csv("data" + os.path.sep + "test.csv")

    # pd.set_option('max_columns', None)
    # print(dataframe_train)


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

    dh.div("Improving performance, c")

    dh.div("Best Models, d")




if __name__ == "__main__":
    main()
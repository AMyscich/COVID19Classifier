# (b.ii) (2 points) Feature exploration. Provide visualizations of the features with respect
# to the outcome (e.g., overlaying histograms, scatter plots), and quantify associations between
# the features and the outcome.

import matplotlib.pyplot as plt
import seaborn as sns

class DataVisualization:

    def __init__(self):
        return

    def dataSeriesSummary(self, dataframe, series):
        return dataframe[series].value_counts()

    def dataSummary(self, dataframe, drop_series=None):

        if drop_series is None:
            drop_series = []

        dataframe = dataframe
        dataframe = dataframe.drop(columns=drop_series)
        for col in dataframe.columns:
            print("\n" + col)
            print(self.dataSeriesSummary(dataframe, col))
        print("\n")
        print(dataframe.count())

        print("\n")
        print(dataframe.describe())

    def boxPlot(self, dataframe, x_label, y_label):

        sns.catplot(x=x_label, y=y_label, #row="class",
                        kind="box", orient="h", height=1.5, aspect=4,
                        data=dataframe)
        plt.title(x_label + " vs. " + y_label)
        plt.show()

    def swarmPlot(self, dataframe, x_label, y_label, class_hue):

        sns.catplot(x=x_label, y=y_label, hue=class_hue,
                    kind="violin", inner="stick", split=True,
                    palette="pastel", data=dataframe)
        plt.title(x_label + " vs. " + y_label)
        plt.show()
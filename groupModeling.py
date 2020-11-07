import pandas as pd
import numpy as np
from preprocess import preprocess
from metrics import metrics


class groupModeling:
    def __init__(self, config, X):
        self.groupModelling = config.groupModelling
        self.metric = config.metric
        self.X = X

    def model(self):
        # average
        if self.groupModelling == 1:
            for group in range(len(self.X)):
                X = self.X[group]
                users = list(X.columns)
                X["avg"] = (X.sum(axis=1) / len(users))
                X.sort_values(by=['avg'], ascending=False)
                recommendations = list(X.index.values)
                print(recommendations)
                # get the error
                '''error = metrics(X)
                if self.metric == 1:
                    error = error.z_recall()
                print(error)

                #movies = list(X.index.values)'''
                # print(X)
                # ids, titles = self.queryXcandidate(self.X[i])
                # puntuations = predict(ids, titles)

        else:
            print("no dataset selected")

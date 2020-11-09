import pandas as pd
import numpy as np
# from preprocess import preprocess
from metrics import metrics
from recommender import recommendationModel


class groupModeling:
    def __init__(self, config, groups):
        self.groupModelling = config.groupModelling
        self.metric = config.metric
        self.groups = groups

    def model(self, tfRecommender):
        # average
        if self.groupModelling == 1:
            totalError = list()
            for group in range(len(self.groups)):
                ids = self.groups[group]
                relMatrix = recommendationModel.predict(ids, tfRecommender)
                # print(relMatrix)
                relMatrix["avg"] = (relMatrix.sum(axis=1) / len(ids))
                relMatrix.sort_values(by=['avg'], ascending=False)
                # print(relMatrix)
                recommendations = list(relMatrix.index.values)
                print("group", group, "recommendations:")
                print(recommendations[:10])
                # get the error
                error = metrics(relMatrix, recommendations[:20])
                if self.metric == 1:
                    groupError = error.zRecall()
                if self.metric == 2:
                    groupError = error.discountedFirstHit()
                if self.metric == 3:
                    groupError = error.normalizedDiscountedCumulativeGain()
                print(groupError)
                totalError.append(groupError)
                print(totalError)
                # movies = list(X.index.values)'''
                # print(X)
                # ids, titles = self.queryXcandidate(self.X[i])
                # puntuations = predict(ids, titles)

        else:
            print("no dataset selected")

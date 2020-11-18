import pandas as pd
import numpy as np
import math


class metrics:
    def __init__(self, X, recommendations):
        self.X = X
        self.recommendations = recommendations

    def getScore(self, metric):
        if metric.lower() == "zrecall":
            groupScore = self.zRecall()
        if metric.lower() == "dfh":
            groupScore = self.discountedFirstHit()
        if metric.lower() == "ndcg":
            groupScore = self.normalizedDiscountedCumulativeGain()
        return groupScore

    def zRecall(self):
        # users that don't have a relevant film in the top-gn
        groupScore = list()
        columns = list(self.X.columns)
        ids = columns[:-1]  # the last one is the score of the row
        for id in ids:
            relevantItems = 0
            i = 0
            while i != len(self.recommendations):
                if self.X.at[self.recommendations[i], id] >= 1.5:  # should be greater than 4 according to recys
                    relevantItems += 1
                i += 1
            recall = relevantItems / i
            if recall == 0:
                userScore = 1
            else:
                userScore = 0
            groupScore.append(userScore)
        return (sum(groupScore) / len(ids))

    def discountedFirstHit(self):
        # relevance in function first position of a relevant movie in the list of top-NG list
        groupScore = list()
        columns = list(self.X.columns)
        ids = columns[:-1]
        for id in ids:
            # self.X[id] = (self.X[id] >= 4).astype(int)
            i = 0
            while i != len(self.recommendations):
                if self.X.at[self.recommendations[i], id] >= 1.5:
                    i += 1
                    break
                i += 1
            if i == len(self.recommendations):
                userScore = 0
            else:
                userScore = 1 / (math.log(i + 1, 2))  # +2 because the rank=i+1
            groupScore.append(userScore)
        return (sum(groupScore) / len(ids))

    def normalizedDiscountedCumulativeGain(self):
        # average
        groupScore = list()
        columns = list(self.X.columns)
        ids = columns[:-1]
        idcg = 0
        for i in range(len(self.recommendations)):
            idcg = idcg + (i + 1 / (math.log(i + 2, 2)))  # +2 i +1 perque comença a 0 i acaba a 19
        for id in ids:
            # self.X[id] = (self.X[id] >= 4).astype(int)
            i = 0
            dcg = 0
            relevantItems = 0
            while i != len(self.recommendations):
                i += 1
                if self.X.at[self.recommendations[i - 1], id] >= 1.5:
                    relevantItems += 1
                    dcg = dcg + (relevantItems / (math.log(i + 1, 2)))
                userScore = dcg / idcg
            groupScore.append(userScore)
        return (sum(groupScore) / len(ids))

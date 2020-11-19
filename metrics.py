import pandas as pd
import numpy as np
import math


class metrics:
    def __init__(self, test, ids, recommendations):
        self.test = test
        self.recommendations = recommendations
        self.ids = ids

    def getScore(self, metric):
        if metric.lower() == "zrecall":
            groupScore = self.zRecall()
        if metric.lower() == "dfh":
            groupScore = self.discountedFirstHit()
        if metric.lower() == "ndcg":
            groupScore = self.normalizedDiscountedCumulativeGain()
        return groupScore

    def recall(self):
        groupScores = list()
        for id in self.ids:
            relevantItems = 0
            i = 0
            while i != len(self.recommendations):
                recommendedMovie = self.recommendations[i]
                # indistinctly of the rating if he/she has view it is relevant
                if ((self.test['user_id'] == id) & (self.test['movie_title'] == recommendedMovie)).any():
                    relevantItems += 1
                i += 1
            userScore = relevantItems / i
            groupScores.append(userScore)
        return groupScores

    def zRecall(self):
        # users that don't have a relevant film in the top-gn
        groupScores = list()
        recalls = self.recall()
        for recall in recalls:
            if recall == 0:
                userScore = 1
            else:
                userScore = 0
            groupScores.append(userScore)
        return groupScores

    def discountedFirstHit(self):
        # relevance in function first position of a relevant movie in the list of top-NG list
        groupScores = list()
        for id in self.ids:
            # self.X[id] = (self.X[id] >= 4).astype(int)
            i = 0
            while i != len(self.recommendations):
                recommendedMovie = self.recommendations[i]
                if ((self.test['user_id'] == id) & (self.test['movie_title'] == recommendedMovie)).any():
                    i += 1
                    break
                i += 1
            if i == len(self.recommendations):
                userScore = 0
            else:
                userScore = 1 / (math.log(i + 1, 2))  # +2 because the rank=i+1
            groupScores.append(userScore)
        return groupScores

    def normalizedDiscountedCumulativeGain(self):
        # average
        groupScores = list()
        idcg = 0
        for i in range(len(self.recommendations)):
            idcg = idcg + (i + 1 / (math.log(i + 2, 2)))  # +2 i +1 perque comença a 0 i acaba a 19
        for id in self.ids:
            # self.X[id] = (self.X[id] >= 4).astype(int)
            i = 0
            dcg = 0
            relevantItems = 0
            while i != len(self.recommendations):
                i += 1
                recommendedMovie = self.recommendations[i]
                if ((self.test['user_id'] == id) & (self.test['movie_title'] == recommendedMovie)).any():
                    relevantItems += 1
                    dcg = dcg + (relevantItems / (math.log(i + 1, 2)))
                userScore = dcg / idcg
            groupScores.append(userScore)
        return groupScores

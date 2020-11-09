import pandas as pd
import numpy as np
import math


class metrics:
    def __init__(self, X, recommendations):
        self.X = X
        self.recommendations = recommendations

    def zRecall(self):
        # proportion of relevant movies in the top-NG list
        groupError = list()
        columns = list(self.X.columns)
        ids = columns[:-1]
        for id in ids:
            # self.X[id] = (self.X[id] >= 4).astype(int)
            relevantItems = 0
            i = 0
            while i != len(self.recommendations):
                if self.X.at[self.recommendations[i], id] >= 1.5:
                    relevantItems += 1
                i += 1
                userError = relevantItems / i  # +2 because the rank=i+1
            groupError.append(userError)
        print(groupError)
        return sum(groupError)

    def discountedFirstHit(self):
        # relevance in function first position of a relevant movie in the list of top-NG list
        groupError = list()
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
                userError = 0
            else:
                userError = 1 / (math.log(i + 1, 2))  # +2 because the rank=i+1
            groupError.append(userError)
        print(groupError)
        return sum(groupError)

    def normalizedDiscountedCumulativeGain(self):
        # average
        groupError = list()
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
                userError = dcg / idcg
            groupError.append(userError)
        print(groupError)
        return sum(groupError)

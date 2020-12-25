import pandas as pd
import numpy as np
import math
import progressbar


class metrics:
    def __init__(self, config, test, groups, recommendations):
        self.test = test
        self.recommendations = recommendations
        self.groups = groups
        self.metric = config.metric
        self.numOfRecommendations = config.numOfRecommendations

    def getScore(self):
        if self.metric.lower() == "zrecall":
            groupScore = self.zRecall()
        if self.metric.lower() == "dfh":
            groupScore = self.discountedFirstHit()
        if self.metric.lower() == "ndcg":
            idcg = 0
            for i in range(self.numOfRecommendations):
                idcg = idcg + (i + 1) / (math.log(i + 2, 10))  # +2 i +1 perque comença a 0 i acaba a 19
            groupScore = self.normalizedDiscountedCumulativeGain(idcg)
        return groupScore

    def zRecall(self):
        groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            tempUsersScores = list()
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            for id in ids:
                seenItems = self.test[self.test['user_id'] == id]
                seenItems = list(seenItems['movie_title'].values)
                matches = list(set(seenItems) & set(recommendations))
                if len(matches) == 0:
                    userScore = 1
                else:
                    userScore = 0
                tempUsersScores.append(userScore)
            groupScores.append((sum(tempUsersScores) / len(ids)))
        totalScore = (sum(groupScores) / len(self.groups))
        bar.finish()
        return totalScore

    def discountedFirstHit(self):
        # relevance in function first position of a relevant movie in the list of top-NG list
        groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            tempUsersScores = list()
            for id in ids:
                positions = []
                seenItems = self.test[self.test['user_id'] == id]
                seenItems = list(seenItems['movie_title'].values)
                matches = list(set(seenItems) & set(recommendations))
                if len(matches) == 0:
                    userScore = 0
                    break
                for match in matches:
                    positions.append(recommendations.index(match))
                userScore = 1 / (math.log(min(positions) + 2, 2))  # +2 because the rank va de 0 a n-1
                tempUsersScores.append(userScore)
            groupScores.append((sum(tempUsersScores) / len(ids)))
        totalScore = (sum(groupScores) / len(self.groups))
        bar.finish()
        return totalScore

    def normalizedDiscountedCumulativeGain(self, idcg):
        # average
        # usersScores = list()
        groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            tempUsersScores = list()
            for id in ids:
                dcg = 0
                seenItems = self.test[self.test['user_id'] == id]
                seenItems = list(seenItems['movie_title'].values)
                matches = list(set(seenItems) & set(recommendations))
                if len(matches) == 0:
                    userScore = 0
                    tempUsersScores.append(userScore)
                    continue
                # print(recommendations)
                # print(matches)
                for match in matches:
                    for i in range(recommendations.index(match), self.numOfRecommendations):
                        dcg = dcg + (1 / (math.log(i + 2, 10)))
                userScore = dcg / idcg
                # print(userScore)
                tempUsersScores.append(userScore)
                # print('-temp-of.group', group, tempUsersScores)
            # usersScores.append(tempUsersScores)
            # print('--users-', usersScores)
            groupScores.append((sum(tempUsersScores) / len(ids)))
            # print('-----groupScores-------', groupScores)
        totalScore = (sum(groupScores) / len(self.groups))
        # print('-------------totalScore-------------------', totalScore)
        bar.finish()
        return totalScore, groupScores

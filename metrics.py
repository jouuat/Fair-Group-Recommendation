import pandas as pd
import numpy as np
import math
import progressbar


class metrics:
    def __init__(self, config, test, groups, recommendations):
        self.test = test
        self.recommendations = recommendations
        self.usersPerGroup = config.usersPerGroup
        self.groups = groups
        self.metric = config.metric
        self.numOfRecommendations = int(config.numOfRecommendations)

    def getScore(self):
        if self.metric.lower() == "zrecall":
            totalScore, groupScores = self.zRecall()
        if self.metric.lower() == "dfh":
            totalScore, groupScores = self.discountedFirstHit()
        if self.metric.lower() == "ndcg":
            idcg = 0
            for i in range(self.numOfRecommendations):
                idcg = idcg + (i + 1) / (math.log(i + 2, 10))  # +2 i +1 perque comença a 0 i acaba a 19
            totalScore, groupScores = self.normalizedDiscountedCumulativeGain(idcg)
        if self.metric.lower() == "ndp":
            idp = 0
            for i in range(self.numOfRecommendations):
                idp = idp + self.usersPerGroup / (math.log(i + 2, 10))  # +2 i +1 perque comença a 0 i acaba a 19
            totalScore, groupScores = self.normalizedDiscountedProportion(idp)
        if self.metric.lower() == "fndcg":
            idcg = 0
            for i in range(self.numOfRecommendations):
                idcg = idcg + (i + 1) / (math.log(i + 2, 10))  # +2 i +1 perque comença a 0 i acaba a 19
            totalScore, groupScores = self.fndcg(idcg)
        if self.metric.lower() == "bndcg":
            idcg = 0
            for i in range(self.numOfRecommendations):
                idcg = idcg + (i + 1) / (math.log(i + 2, 10))  # +2 i +1 perque comença a 0 i acaba a 19
            totalScore, groupScores = self.bndcg(idcg)
        if self.metric.lower() == "ourmetric2":
            idcg = 0
            for i in range(self.numOfRecommendations):
                idcg = idcg + (i + 1) / (math.log(i + 2, 10))  # +2 i +1 perque comença a 0 i acaba a 19
            totalScore, groupScores = self.ourMetric2(idcg)
        return totalScore, groupScores

    def zRecall(self):
        groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            tempUsersScores = list()
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            recommendations = recommendations[:self.numOfRecommendations]
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
        return totalScore, groupScores

    def discountedFirstHit(self):
        # relevance in function first position of a relevant movie in the list of top-NG list
        groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            recommendations = recommendations[:self.numOfRecommendations]
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
        return totalScore, groupScores

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
            recommendations = recommendations[:self.numOfRecommendations]
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

    def normalizedDiscountedProportion(self, idp):
        # average
        # usersScores = list()
        groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            recommendations = recommendations[:self.numOfRecommendations]
            groupScore = 0
            dp = 0
            i = 0
            for recommendation in recommendations:
                i += 1
                proportion = 0
                for id in ids:
                    seenItems = self.test[self.test['user_id'] == id]
                    seenItems = list(seenItems['movie_title'].values)
                    if recommendation in seenItems:
                        proportion += 1
                    else:
                        continue
                dp = dp + (proportion / (math.log(i + 1, 10)))
                groupScore = dp / idp
            groupScores.append(groupScore)
            # print('-----groupScores-------', groupScores)
        totalScore = (sum(groupScores) / len(self.groups))
        # print('-------------totalScore-------------------', totalScore)
        bar.finish()
        return totalScore, groupScores

    def fndcg(self, idcg):
        # average
        # usersScores = list()
        groupAccuracy = []
        groupFairness = []
        maxNDCG = 1
        # groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            recommendations = recommendations[:self.numOfRecommendations]
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
                for match in matches:
                    for i in range(recommendations.index(match), self.numOfRecommendations):
                        dcg = dcg + (1 / (math.log(i + 2, 10)))
                userScore = dcg / idcg
                tempUsersScores.append(userScore)
            var_ = np.var(tempUsersScores)
            mean_ = sum(tempUsersScores) / len(tempUsersScores)
            if mean_ == 0:
                # groupScore = 0
                fairness = 0
            else:
                # maxScore = mean_ * len(tempUsersScores)
                # floor = (maxScore // maxNDCG)
                # worstVar = (floor * (maxNDCG - mean_) ** 2 + ((maxScore - floor * maxNDCG) - mean_) ** 2 + (len(tempUsersScores) - floor - 1) * (0 - mean_) ** 2) / len(tempUsersScores)
                floor = (len(tempUsersScores) // 2)
                worstMean = (floor * maxNDCG) / len(tempUsersScores)
                worstVar = (floor * (maxNDCG - worstMean) ** 2 + (floor + 1) * (0 - worstMean) ** 2) / len(tempUsersScores)
                fairness = (worstVar - var_) / worstVar
                # fairness = var_
            groupAccuracy.append(mean_)
            groupFairness.append(fairness)
        totalAccuracy = (sum(groupAccuracy) / len(self.groups))
        totalFairness = (sum(groupFairness) / len(self.groups))
        # print('-------------totalScore-------------------', totalScore)
        bar.finish()
        return totalAccuracy, totalFairness

    def bndcg(self, idcg):
        # average
        # usersScores = list()
        m = 1
        p = 1
        groupScores = []
        # groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            recommendations = recommendations[:self.numOfRecommendations]
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
                for match in matches:
                    for i in range(recommendations.index(match), self.numOfRecommendations):
                        dcg = dcg + (1 / (math.log(i + 2, 10)))
                userScore = dcg / idcg
                tempUsersScores.append(userScore)
            tempUsersScores.sort(reverse=True)
            weights = [m * ((i + 1) ** p) for i in range(len(ids))]
            indScores = [a * b for a, b in zip(tempUsersScores, weights)]
            groupScore = sum(indScores) / sum(weights)
            groupScores.append(groupScore)
            # print('-----groupScores-------', groupScores)
        totalScore = (sum(groupScores) / len(self.groups))
        # print('-------------totalScore-------------------', totalScore)
        bar.finish()
        return totalScore, groupScores

    def ourMetric2(self, idcg):
        # average
        # usersScores = list()
        groupScores = []
        maxNDCG = 1
        # groupScores = list()
        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]["members"]
            recommendations = self.recommendations[group]["recommendations"]
            recommendations = recommendations[:self.numOfRecommendations]
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
                for match in matches:
                    for i in range(recommendations.index(match), self.numOfRecommendations):
                        dcg = dcg + (1 / (math.log(i + 2, 10)))
                userScore = dcg / idcg
                tempUsersScores.append(userScore)
            var_ = np.var(tempUsersScores)
            mean_ = sum(tempUsersScores) / len(tempUsersScores)
            min_ = min(tempUsersScores)
            max_ = max(tempUsersScores)
            if mean_ == 0:
                groupScore = 0
            else:
                maxScore = mean_ * len(tempUsersScores)
                floor = (maxScore // maxNDCG)
                worstVar = (floor * (maxNDCG - mean_) ** 2 + ((maxScore - floor * maxNDCG) - mean_) ** 2 + (len(tempUsersScores) - floor - 1) * (0 - mean_) ** 2) / len(tempUsersScores)
                groupScore = min_ + (max_ - min_) * ((worstVar - var_) / worstVar)
            groupScores.append(groupScore)
        totalScore = (sum(groupScores) / len(self.groups))
        # print('-------------totalScore-------------------', totalScore)
        bar.finish()
        return totalScore, groupScores

import pandas as pd
import numpy as np
# from preprocess import preprocess
from metrics import metrics
from recommender import recommendationModel


class groupModeling:
    def __init__(self, config, groups):
        self.groupModelling = config.groupModelling
        self.numOfRecommendations = config.numOfRecommendations
        self.metric = config.metric
        self.groups = groups

    def model(self, tfRecommender):
        # AVERAGE
        totalScores = list()
        if self.groupModelling.lower() == "average":
            for group in range(len(self.groups)):
                ids = self.groups[group]
                relMatrix = recommendationModel.predict(ids, tfRecommender)
                # print(relMatrix)
                relMatrix["score"] = (relMatrix.sum(axis=1) / len(ids))
                relMatrix.sort_values(by=['score'], ascending=False)
                # print(relMatrix)
                recommendations = list(relMatrix.index.values)
                # get the error
                metrics_ = metrics(relMatrix, recommendations[:20])
                groupScore = metrics_.getScore(self.metric)
                # print ("with a score of", groupScore)
                totalScores.append(groupScore)
            totalScore = (sum(totalScores) / len(self.groups))
            # print("And a total score of:", totalScore)

        # GFAR
        if self.groupModelling.lower() == "gfar":
            for group in range(len(self.groups)):
                ids = self.groups[group]
                relMatrix = recommendationModel.predict(ids, tfRecommender)
                # get transform the relevance scores to (20-rank)/20
                gfarMatrix = relMatrix
                for id in range(len(ids)):
                    gfarMatrix = gfarMatrix.sort_values(by=ids[id], ascending=False, inplace=False)
                    for rank in range(len(gfarMatrix.index)):
                        if rank < self.numOfRecommendations:
                            gfarMatrix.iloc[rank, id] = (self.numOfRecommendations - rank - 1) / self.numOfRecommendations
                        else:
                            gfarMatrix.iloc[rank, id] = 0
                recommendations = list()
                for rank in range(self.numOfRecommendations):
                    gfarMatrix["score"] = (gfarMatrix.sum(axis=1) / len(ids))
                    gfarMatrix = gfarMatrix.sort_values(by=['score'], ascending=False, inplace=False)
                    recommendations.append(gfarMatrix.index[0])
                    _ = 1 - gfarMatrix.iloc[0]
                    gfarMatrix = gfarMatrix.mul(_, axis=1)
                    gfarMatrix = gfarMatrix[1:]
                    gfarMatrix.drop(columns=['score'])
                # print("group", group, "recommendations:")
                # print(recommendations[:5])
                # get the error
                metrics_ = metrics(relMatrix, recommendations[:20])
                groupScore = metrics_.getScore(self.metric)
                # print ("with a score of", groupScore)
                totalScores.append(groupScore)
            totalScore = (sum(totalScores) / len(self.groups))
            # print("And a total score of:", totalScore)
        self.recommendationsInfo(totalScore, recommendations)

    def recommendationsInfo(self, totalScore, recommendations):
        print("recommendations for", len(self.groups), " groups have been created using the", self.groupModelling, "techinique,")
        print("and obtained a socore of", totalScore, "with", self.metric, "metric\n")

        print("the 5 first recommendations for the last group are:", recommendations[:5])

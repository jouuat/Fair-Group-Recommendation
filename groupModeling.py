import pandas as pd
import numpy as np
# from preprocess import preprocess
from metrics import metrics
from recommender import recommendationModel
from reputation import reputation
import math


class groupModeling:
    def __init__(self, config, groups, ratings_pd):
        self.groupModelling = config.groupModelling
        self.numOfRecommendations = config.numOfRecommendations
        self.metric = config.metric
        self.groups = groups
        self.numOfIterations = config.numOfIterations
        # for reputation
        self.ratings_pd = ratings_pd
        self.greedylmlambda = config.greedylmlambda
        self.p = config.p

    def model(self, tfRecommender):
        # AVERAGE
        sizeDataset = len(self.ratings_pd["user_rating"].tolist())
        train = self.ratings_pd.head(int(sizeDataset * 0.8))
        test = self.ratings_pd.tail(int(sizeDataset * 0.2))
        totalScores = list()
        recommendations = list()

        for group in range(len(self.groups)):
            ids = self.groups[group]
            relMatrix = recommendationModel.predict(ids, tfRecommender)

            # AVERAGE
            if self.groupModelling.lower() == "average":
                relMatrix["score"] = (relMatrix.sum(axis=1) / len(ids))
                relMatrix.sort_values(by=['score'], ascending=False)
                recommendations = list(relMatrix.index.values)

            # GFAR
            if self.groupModelling.lower() == "gfar":
                gfarMatrix = relMatrix
                for id in range(len(ids)):
                    gfarMatrix = gfarMatrix.sort_values(by=ids[id], ascending=False, inplace=False)
                    for rank in range(len(gfarMatrix.index)):
                        if rank < self.numOfRecommendations:
                            gfarMatrix.iloc[rank, id] = (self.numOfRecommendations - rank - 1) / self.numOfRecommendations
                        else:
                            gfarMatrix.iloc[rank, id] = 0
                for rank in range(self.numOfRecommendations):
                    gfarMatrix["score"] = (gfarMatrix.sum(axis=1) / len(ids))
                    gfarMatrix = gfarMatrix.sort_values(by=['score'], ascending=False, inplace=False)
                    recommendations.append(gfarMatrix.index[0])
                    _ = 1 - gfarMatrix.iloc[0]
                    gfarMatrix = gfarMatrix.mul(_, axis=1)
                    gfarMatrix = gfarMatrix[1:]
                    gfarMatrix.drop(columns=['score'])

            # GREEDYLM
            if self.groupModelling.lower() == "greedylm":
                bestCost = -math.inf
                bestRecommendation = 0
                iteration = 0
                utility = [0] * len(ids)
                print(utility)
                rank = 0
                movies = relMatrix.index.tolist()
                trainedMovies = list()
                while rank <= self.numOfRecommendations:
                    # print(utility)
                    newRecommendation = np.random.choice(movies, size=1, replace=False).item(0)
                    trainedMovies.append(newRecommendation)
                    movies.remove(newRecommendation)
                    newUtility = reputation().greedylm(newRecommendation, relMatrix, utility, len(recommendations))
                    SocialWelfare = sum(utility) / len(utility)
                    Fairness = min(newUtility)
                    newCost = self.greedylmlambda * SocialWelfare + (1 - self.greedylmlambda) * Fairness
                    if newCost > bestCost:
                        bestCost = newCost
                        bestRecommendation = newRecommendation
                        bestUtility = newUtility
                        # print('new recommendation', bestRecommendation, 'with cost', bestCost, 'rank', rank, 'with utility',
                        #      bestUtility, 'previous recommendations', recommendations, '---------------------------------')
                        iteration = 0
                    if iteration >= self.numOfIterations:
                        iteration = 0
                        bestCost = -math.inf
                        rank += 1
                        utility = bestUtility
                        recommendations.append(bestRecommendation)
                        print('trainedMovies:', trainedMovies, 'bestRecommendation', bestRecommendation)
                        trainedMovies.remove(bestRecommendation)
                        movies = movies + trainedMovies
                    iteration += 1

            # get the error
            # print(recommendations)
            metrics_ = metrics(test, ids, recommendations[:20])
            groupScores = metrics_.getScore(self.metric)
            totalScores.append((sum(groupScores) / len(ids)))
        totalScore = (sum(totalScores) / len(self.groups))
        self.recommendationsInfo(totalScore, recommendations)

    def recommendationsInfo(self, totalScore, recommendations):
        print("recommendations for", len(self.groups), " groups have been created using the", self.groupModelling, "techinique,")
        print("and obtained a socore of", totalScore, "with", self.metric, "metric\n")

        print("the 5 first recommendations for the last group are:", recommendations[:5])

import pandas as pd
import numpy as np
# from preprocess import preprocess
from metrics import metrics
from recommender import recommendationModel
from reputation import reputation
import math
import progressbar


class groupModeling:
    def __init__(self, config, groups, ratings_pd):
        self.groupModelling = config.groupModelling
        self.numOfRecommendations = config.numOfRecommendations
        self.metric = config.metric
        self.groups = groups
        self.numOfIterations = config.numOfIterations
        self.usersPerGroup = config.usersPerGroup
        # for reputation
        self.ratings_pd = ratings_pd
        self.greedylmlambda = config.greedylmlambda
        self.p = config.p
        self.mproportionality = config.mproportionality
        self.numRelevantItems = config.numRelevantItems
        if str(self.numRelevantItems).lower() == "auto":
            self.numRelevantItems = self.numOfRecommendations

    def model(self, tfRecommender):
        # AVERAGE
        sizeDataset = len(self.ratings_pd["user_rating"].tolist())
        train = self.ratings_pd.head(int(sizeDataset * 0.8))
        test = self.ratings_pd.tail(int(sizeDataset * 0.2))
        totalScores = list()
        recommendations = list()

        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()

        for group in range(len(self.groups)):
            bar.update(group + 1)
            ids = self.groups[group]
            relMatrix = recommendationModel.predict(ids, tfRecommender)
            # Penalize those films already seen by a user ( TODO MASSA LENT)
            '''for id in relMatrix.columns:
                minPuntuation = min(relMatrix[id])
                for item in relMatrix.index:
                    if ((train['user_id'] == id) & (train['movie_title'] == item)).any():
                        print('id:', id, 'and item', item)
                        relMatrix.loc[item, id] = minPuntuation
            print(relMatrix)'''

            # ----------------------------------- AVERAGE -----------------------------------
            if self.groupModelling.lower() == "average":
                relMatrix["score"] = (relMatrix.sum(axis=1) / len(ids))
                relMatrix.sort_values(by=['score'], ascending=False)
                recommendations = list(relMatrix.index.values[:self.numOfRecommendations])

            # ------------------------------------ GFAR -------------------------------------
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

            # -------------------------------- GREEDYLM ---------------------------------
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

            # -------------------------------- SPGREEDY ------------------------------
            if self.groupModelling.lower() == "spgreedy":
                bestCost = -math.inf
                bestRecommendation = 0
                iteration = 0
                rank = 0
                movies = relMatrix.index.tolist()
                trainedMovies = list()
                satUsers = list()
                # crete a dataframe with each user relevant films
                for i in range(len(ids)):
                    relMatrix.sort_values(by=[ids[i]], ascending=False)
                    userRelItems = list(relMatrix.index.values)[:self.numRelevantItems]
                    if i == 0:
                        relevantItems = pd.DataFrame(userRelItems, columns=[ids[i]])
                        continue
                    relevantItems[ids[i]] = userRelItems
                while rank <= self.numOfRecommendations:
                    newRecommendation = np.random.choice(movies, size=1, replace=False).item(0)
                    trainedMovies.append(newRecommendation)
                    movies.remove(newRecommendation)
                    tempSatUsers = satUsers
                    for i in range(len(ids)):
                        if newRecommendation in relevantItems[ids[i]]:
                            tempSatUsers.append(ids[i])
                    tempSatUsers = list(set(tempSatUsers))
                    newCost = len(set(tempSatUsers) - set(satUsers))
                    if newCost > bestCost:
                        bestCost = newCost
                        bestRecommendation = newRecommendation
                        bestSatUsers = tempSatUsers
                        # print('new recommendation', bestRecommendation, 'with cost', bestCost, 'rank', rank, 'with utility', bestTempSatUsers, 'previous recommendations', recommendations, '---------------------------------')
                        iteration = 0
                    if iteration >= self.numOfIterations:
                        iteration = 0
                        bestCost = -math.inf
                        rank += 1
                        satUsers = bestSatUsers
                        recommendations.append(bestRecommendation)
                        trainedMovies.remove(bestRecommendation)
                        movies = movies + trainedMovies
                    iteration += 1

            # ------------------------------- FAI -----------------------------------
            if self.groupModelling.lower() == "fai":
                rank = 0
                stack = []
                # recommendationsPerUser = math.ceil(self.numOfRecommendations / self.usersPerGroup)
                while rank <= self.numOfRecommendations:
                    if math.floor(rank / self.usersPerGroup) == 0:
                        availableIds = list(set(ids) - set(stack))
                        bestRel = -math.inf
                        for id in availableIds:
                            column = pd.to_numeric(relMatrix[id])
                            maxFilm = column.idxmax()
                            maxRel = relMatrix.loc[maxFilm, id]
                            if maxRel > bestRel:
                                bestRel = maxRel
                                bestFilm = maxFilm
                                bestId = id
                        stack.append(bestId)
                        print('stack', stack)
                        recommendations.append(bestFilm)
                        rank += 1
                    elif (math.floor(rank / self.usersPerGroup) % 2) == 0:
                        for id in stack:
                            print('id non reversed', id)
                            column = pd.to_numeric(relMatrix[id])
                            bestFilm = column.idxmax()
                            recommendations.append(bestFilm)
                            rank += 1
                    else:
                        for id in reversed(stack):
                            print('id reversed', id)
                            column = pd.to_numeric(relMatrix[id])
                            bestFilm = column.idxmax()
                            recommendations.append(bestFilm)
                            rank += 1

            # ------------------------------- XPO --------------------------------
            if self.groupModelling.lower() == "xpo":
                # get the N - level Pareto
                usersTopN = list()
                for id in ids:
                    sorted = relMatrix.sort_values(by=[id], ascending=False)
                    userTopN = list(sorted.head(self.numOfRecommendations).index)
                    if len(usersTopN) == 0:
                        usersTopN = userTopN
                        continue
                    usersTopN = usersTopN + userTopN
                usersTopN = list(set(usersTopN))  # remove duplications
                # print('usersTopN', list(usersTopN))
                _ = relMatrix[ids]
                xpoMatrix = _.loc[list(usersTopN), :]
                # print(xpoMatrix)
                # create random weights
                filmCounts = pd.DataFrame(0, columns=["counts"], index=xpoMatrix.index)
                for i in range(self.numOfIterations):
                    weights = list(np.random.random(size=self.usersPerGroup))
                    weights /= sum(weights)
                    weightedMatrix = xpoMatrix
                    # print(weightedMatrix)
                    if i != 0:
                        weightedMatrix = weightedMatrix.drop(columns=["score"])
                        # print(weightedMatrix)
                    weightedMatrix.mul(weights, axis=1)
                    weightedMatrix["score"] = (weightedMatrix.sum(axis=1) / len(ids))
                    weightedMatrix.sort_values(by=['score'], ascending=False)
                    # print('after sort', weightedMatrix)
                    weightedRecommendations = list(relMatrix.index.values)[:self.numOfRecommendations]
                    for recommendation in weightedRecommendations:
                        filmCounts.loc[recommendation, :] += 1
                    # print("iteration", i, "filmCounts", filmCounts)
                filmCounts.sort_values(by=['counts'], ascending=False)
                recommendations = list(relMatrix.index.values)[:self.numOfRecommendations]

            # get the error
            # print(recommendations)
            metrics_ = metrics(test, ids, recommendations[:20])
            groupScores = metrics_.getScore(self.metric)
            totalScores.append((sum(groupScores) / len(ids)))
        totalScore = (sum(totalScores) / len(self.groups))
        bar.finish()
        self.recommendationsInfo(totalScore, recommendations)

    def recommendationsInfo(self, totalScore, recommendations):
        print("recommendations for", len(self.groups), " groups have been created using the", self.groupModelling, "techinique,")
        print("and obtained a socore of", totalScore, "with", self.metric, "metric\n")

        print("the 5 first recommendations for the last group are:", recommendations[:5])

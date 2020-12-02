import pandas as pd
import numpy as np
# from preprocess import preprocess
from metrics import metrics
from recommender import recommendationModel
from reputation import reputation
import math
import progressbar
import time


class groupModeling:
    def __init__(self, config, groups, ratings_pd):
        self.groupModeling = config.groupModeling
        self.numOfRecommendations = config.numOfRecommendations
        self.metric = config.metric
        self.groups = groups
        self.numOfIterations = config.numOfIterations
        self.usersPerGroup = config.usersPerGroup
        # for reputation
        self.ratings_pd = ratings_pd
        self.lambdaGreedylm = config.lambdaGreedylm
        self.p = config.p
        self.mproportionality = config.mproportionality
        self.numRelevantItems = config.numRelevantItems
        self.all = config.all
        if str(self.numRelevantItems).lower() == "auto":
            self.numRelevantItems = self.numOfRecommendations
        self.nroot = config.nroot
        self.scale = config.scale
        self.lambdaOur = config.lambdaOur
        self.lambdaReputation = config.lambdaReputation

    def model(self, tfRecommender):
        # AVERAGE
        sizeDataset = len(self.ratings_pd["user_rating"].tolist())
        train = self.ratings_pd.head(int(sizeDataset * 0.8))
        test = self.ratings_pd.tail(int(sizeDataset * 0.2))
        totalScores = list()

        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        start_time = time.time()

        for group in range(len(self.groups)):
            recommendations = list()
            bar.update(group + 1)
            ids = self.groups[group]
            '''GET IDS OF USERS THAT HAVE SIMILAR RELEVANCES

            ids = list()
            unique_users = self.ratings_pd["user_id"].unique().tolist()
            refUserId = np.random.choice(unique_users, replace=False)
            i = 0
            tries = 0
            while i != self.usersPerGroup:
                candidateUserId = np.random.choice(unique_users, replace=False)
                tempIds = [refUserId, candidateUserId]
                tempMatrix = recommendationModel.predict(tempIds, tfRecommender)
                if (max(tempMatrix[candidateUserId]) > 0.9 * max(tempMatrix[refUserId])) & (max(tempMatrix[candidateUserId]) < 1.1 * max(tempMatrix[refUserId])):
                    ids.append(candidateUserId)
                    i += 1
                    tries = 0
                if tries == 150:
                    ids = list()
                    tries = 0
                    i = 0
                    refUserId = np.random.choice(unique_users, replace=False)
                tries += 1

            END OF SIMILAR RELEVANCES '''
            relMatrix = recommendationModel.predict(ids, tfRecommender)
            # Penalize those films already seen by a user
            for id in ids:
                _ = train[train['user_id'] == id]
                seenItems = _['movie_title']
                minPuntuation = min(relMatrix[id])
                for seenItem in seenItems:
                    # print('id:', id, 'and item', seenItem)
                    relMatrix.loc[seenItem, id] = minPuntuation

            # scale relMatrix -> all the values between 0 and 1
            if self.scale:
                relMatrix -= relMatrix.min()
                relMatrix /= relMatrix.max()

            # ----------------------------------- AVERAGE -----------------------------------
            if self.groupModeling.lower() == "average":
                avgMatrix = relMatrix
                avgMatrix["score"] = (avgMatrix.sum(axis=1) / len(ids))
                avgMatrix.sort_values(by=['score'], ascending=False)
                recommendations = list(avgMatrix.index.values[:self.numOfRecommendations])

            # ------------------------------------ GFAR -------------------------------------
            if self.groupModeling.lower() == "gfar":
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
            if self.groupModeling.lower() == "greedylm":
                bestCost = -math.inf
                bestRecommendation = 0
                iteration = 0
                utility = [0] * len(ids)
                # print(utility)
                rank = 0
                movies = relMatrix.index.tolist()
                trainedMovies = list()
                while rank <= self.numOfRecommendations:
                    # print(utility)
                    lmMatrix = relMatrix
                    newRecommendation = np.random.choice(movies, size=1, replace=False).item(0)
                    trainedMovies.append(newRecommendation)
                    movies.remove(newRecommendation)
                    newUtility = reputation().greedylm(newRecommendation, lmMatrix, utility, len(recommendations))
                    SocialWelfare = sum(utility) / len(utility)
                    Fairness = min(newUtility)
                    newCost = self.lambdaGreedylm * SocialWelfare + (1 - self.lambdaGreedylm) * Fairness
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
                        # print('trainedMovies:', trainedMovies, 'bestRecommendation', bestRecommendation)
                        trainedMovies.remove(bestRecommendation)
                        movies = movies + trainedMovies
                    iteration += 1

            # -------------------------------- SPGREEDY ------------------------------
            if self.groupModeling.lower() == "spgreedy":
                bestCost = -math.inf
                bestRecommendation = 0
                iteration = 0
                rank = 0
                movies = relMatrix.index.tolist()
                trainedMovies = list()
                satUsers = list()
                # crete a dataframe with each user relevant films
                for i in range(len(ids)):
                    spMatrix = relMatrix
                    spMatrix.sort_values(by=[ids[i]], ascending=False)
                    userRelItems = list(spMatrix.index.values)[:self.numRelevantItems]
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
            if self.groupModeling.lower() == "fai":
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
                        # print('stack', stack)
                        recommendations.append(bestFilm)
                        rank += 1
                    elif (math.floor(rank / self.usersPerGroup) % 2) == 0:
                        for id in stack:
                            # print('id non reversed', id)
                            column = pd.to_numeric(relMatrix[id])
                            bestFilm = column.idxmax()
                            recommendations.append(bestFilm)
                            rank += 1
                    else:
                        for id in reversed(stack):
                            # print('id reversed', id)
                            column = pd.to_numeric(relMatrix[id])
                            bestFilm = column.idxmax()
                            recommendations.append(bestFilm)
                            rank += 1

            # ------------------------------- XPO --------------------------------
            if self.groupModeling.lower() == "xpo":
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
                xpoMatrix = _.loc[list(usersTopN), :]  # agafem la unio dels Top-N individuals
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
                    weightedRecommendations = list(weightedMatrix.index.values)[:self.numOfRecommendations]
                    for recommendation in weightedRecommendations:
                        filmCounts.loc[recommendation, :] += 1
                    # print("iteration", i, "filmCounts", filmCounts)
                filmCounts.sort_values(by=['counts'], ascending=False)
                recommendations = list(filmCounts.index.values)[:self.numOfRecommendations]

            # ------------------------------- OUR --------------------------------
            if self.groupModeling.lower() == "our":
                bestCost = math.inf
                bestRecommendation = 0
                iteration = 0
                rank = 0
                movies = relMatrix.index.tolist()
                trainedMovies = list()
                satUsers = [0] * len(ids)
                # crete a dataframe with each user relevant films
                for i in range(len(ids)):
                    relMatrix.sort_values(by=[ids[i]], ascending=False)
                    userRelItems = list(relMatrix.index.values)[:self.numRelevantItems]
                    if i == 0:
                        relevantItems = pd.DataFrame(userRelItems, columns=[ids[i]])
                        continue
                    relevantItems[ids[i]] = userRelItems
                tempRelMatrix = relMatrix
                while rank <= self.numOfRecommendations:
                    newRecommendation = np.random.choice(movies, size=1, replace=False).item(0)
                    trainedMovies.append(newRecommendation)
                    movies.remove(newRecommendation)
                    tempSatUsers = satUsers
                    for i in range(len(ids)):
                        if newRecommendation in relevantItems[ids[i]]:
                            tempSatUsers[i] = 0
                        else:
                            tempSatUsers[i] = tempSatUsers[i] + max(tempRelMatrix[ids[i]])
                    # newCost = sum(tempSatUsers) * (np.var(tempRelMatrix.loc[newRecommendation, :]) ** (1 / float(self.nroot)))
                    newCost = ((1 - self.lambdaOur) * sum(tempSatUsers)) + (self.lambdaOur * np.var(tempRelMatrix.loc[newRecommendation, :]))
                    # newCost = ((1 - self.lambdaOur) * sum(tempSatUsers)) + (self.lambdaOur * (1 - min(tempRelMatrix.loc[newRecommendation, :])))
                    if newCost < bestCost:
                        bestCost = newCost
                        bestRecommendation = newRecommendation
                        bestSatUsers = tempSatUsers
                        # print('new recommendation', bestRecommendation, 'with cost', bestCost, 'rank', rank, 'with utility',
                        #       bestSatUsers, 'previous recommendations', recommendations, '---------------------------------')
                        iteration = 0
                    if iteration >= self.numOfIterations:
                        # print('relMatrix', relMatrix, 'bestRecommendation', bestRecommendation)
                        iteration = 0
                        bestCost = + math.inf
                        rank += 1
                        satUsers = bestSatUsers
                        recommendations.append(bestRecommendation)
                        trainedMovies.remove(bestRecommendation)
                        movies = movies + trainedMovies
                        tempRelMatrix.drop(tempRelMatrix.loc[tempRelMatrix.index == bestRecommendation].index)  # otherwise will appear the maximum value
                    iteration += 1

            # ------------------------------- OUR 2 --------------------------------
            if self.groupModeling.lower() == "our2":
                usersTopN = list()
                i = 0
                for id in ids:
                    idRelMatrix = relMatrix[id]
                    _ = idRelMatrix.sort_values(ascending=False)
                    sorted = _[:math.ceil(1.3 * self.numOfRecommendations)]
                    for rank in range(len(sorted.index)):
                        # first +2 because rank starts with 0 not 1, seond + 2 because is the enxt value of rank +1
                        sorted.iloc[rank] = (1 / (math.log(rank + 2, 2))) - (1 / (math.log(len(sorted.index) + 2, 2)))
                    if i == 0:
                        ourMatrix = sorted.to_frame()
                        initialRight = sorted.sum()
                        i = 1
                        continue
                    ourMatrix = pd.concat([ourMatrix, sorted], axis=1, sort=False)
                    # ourMatrix = pd.merge(ourMatrix, idRelMatrix, how='outer')
                ourMatrix = ourMatrix.fillna(0)
                bestRight = [- initialRight] * self.usersPerGroup
                for rank in range(self.numOfRecommendations):
                    rights = ourMatrix + bestRight
                    rights["score"] = rights.sum(axis=1)
                    rights = rights.sort_values(by=['score'], ascending=False, inplace=False)
                    rights = rights.drop(columns=['score'])
                    recommendation = rights.index[0]
                    bestRight = list(rights.iloc[0])  # change the sign because the obtained is negative
                    recommendations.append(recommendation)
                    ourMatrix = ourMatrix.drop(ourMatrix.loc[ourMatrix.index == recommendation].index)

            # ------------------------------- REPUTATION --------------------------------
            if self.groupModeling.lower() == "reputation":
                rank = 0
                movies = relMatrix.index.tolist()
                reputation_ = [1] * self.usersPerGroup
                usersRatedItems = list()
                reputationMatrix = relMatrix
                # Get the user real preditctions and the group relevances
                for id in ids:
                    _ = train[train['user_id'] == id]
                    userRatedItems = _[['movie_title', 'user_rating']]
                    groupRel = relMatrix.loc[list(userRatedItems['movie_title']), relMatrix.columns != id]
                    userRatedItems = userRatedItems.set_index("movie_title")
                    userRatedItems = userRatedItems.join(groupRel, how="inner")
                    usersRatedItems.append(userRatedItems)
                while rank <= self.numOfRecommendations:
                    reputation_ = reputation().reputationBased(self.lambdaReputation, usersRatedItems, reputation_, self.p)
                    weightedMatrix = reputationMatrix.mul(reputation_, axis=1)  # / sum(reputation_) not necessary to divide since all of them are divided by the same num
                    weightedMatrix["score"] = weightedMatrix.sum(axis=1)
                    weightedMatrix = weightedMatrix.sort_values(by=['score'], ascending=False, inplace=False)
                    recommendation = weightedMatrix.index[0]
                    weightedMatrix = weightedMatrix.drop(columns=['score'])
                    recommendations.append(recommendation)
                    reputationMatrix = reputationMatrix.drop(reputationMatrix.loc[reputationMatrix.index == recommendation].index)
                    rank += 1

            # get the error
            # print(recommendations)
            metrics_ = metrics(test, ids, recommendations)
            groupScores = metrics_.getScore(self.metric)
            totalScores.append((sum(groupScores) / len(ids)))
        totalScore = (sum(totalScores) / len(self.groups))
        bar.finish()
        # your script
        elapsed_time = time.time() - start_time
        print("elapsed time", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)))
        if not self.all:
            self.recommendationsInfo(totalScore, recommendations)
        return totalScore

    def recommendationsInfo(self, totalScore, recommendations):
        print("recommendations for", len(self.groups), " groups have been created using the", self.groupModeling, "technique,")
        print("and obtained a socore of", totalScore, "with", self.metric, "metric\n")

        print("the 5 first recommendations for the last group are:", recommendations[:5])

import pandas as pd
import numpy as np
# from preprocess import preprocess
from metrics import metrics
from recommender import recommendationModel
from reputation import reputation
import math
import progressbar
import time
import json
from collections import Counter


class groupModeling:
    def __init__(self, config, groups, train, path):
        self.groupModeling = config.groupModeling
        self.numOfRecommendations = config.numOfRecommendations
        self.metric = config.metric
        self.groups = groups
        self.numOfIterations = config.numOfIterations
        self.usersPerGroup = config.usersPerGroup
        # for reputation
        self.train = train
        self.lambdaGreedylm = config.lambdaGreedylm
        self.p = config.p
        self.mproportionality = config.mproportionality
        self.numRelevantItems = config.numRelevantItems
        if str(self.numRelevantItems).lower() == "auto":
            self.numRelevantItems = self.numOfRecommendations
        self.nroot = config.nroot
        self.scale = config.scale
        self.lambdaOur = config.lambdaOur
        self.lambdaReputation = config.lambdaReputation
        self.path = path

    def model(self, tfRecommender):
        # AVERAGE
        sizeDataset = len(self.train["user_rating"].tolist())
        train = self.train.head(int(sizeDataset * 0.8))
        allRecommnedations = []

        bar = progressbar.ProgressBar(maxval=len(self.groups), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        start_time = time.time()

        for group in range(len(self.groups)):
            recommendations = list()
            bar.update(group + 1)
            ids = self.groups[group]['members']
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
                rank = 0
                lmMatrix = relMatrix
                utility = lmMatrix  # not necessary to initialize the numbers because at the initial step will be mutiply it by 0
                while rank <= self.numOfRecommendations:
                    utility = reputation().greedylm(lmMatrix, utility, len(recommendations))
                    utility["score"] = self.lambdaGreedylm * (utility.sum(axis=1) / self.usersPerGroup) + (1 - self.lambdaGreedylm) * utility.min(axis=1)
                    utility = utility.sort_values(by=['score'], ascending=False, inplace=False)
                    recommendation = utility.index[0]
                    utility = utility.drop(columns=['score'])
                    recommendations.append(recommendation)
                    utility = utility.drop([recommendation])
                    lmMatrix = lmMatrix.drop([recommendation])
                    rank += 1

            # -------------------------------- SPGREEDY ------------------------------
            if self.groupModeling.lower() == "spgreedy":
                rank = 0
                satisfiedUsers = list()
                individualRecom = []
                # crete a dataframe with each user relevant films
                spMatrix = relMatrix
                for i in range(len(ids)):
                    spMatrix = spMatrix.sort_values(by=[ids[i]], ascending=False, inplace=False)
                    userRelItems = list(spMatrix.index.values)[:self.numRelevantItems]
                    individualRecom.append(userRelItems)

                while rank < self.numOfRecommendations:
                    users = range(self.usersPerGroup)
                    unSatisfiedUsers = list(set(users) - set(satisfiedUsers))
                    consideredRecommnendations = [individualRecom[i] for i in unSatisfiedUsers]
                    indTogether = [item for sublist in consideredRecommnendations for item in sublist]
                    itemAppereances = dict(Counter(indTogether))
                    orderedItemAppereances = sorted(itemAppereances.items(), key=lambda item: item[1], reverse=True)
                    recommendation = orderedItemAppereances[0][0]
                    recommendations.append(recommendation)
                    rank += 1
                    # delete the selected movie in each and write the satisfiedUsers
                    userNum = 0
                    for userRecommendations in individualRecom:
                        if recommendation in userRecommendations:
                            userRecommendations.remove(recommendation)
                            satisfiedUsers.append(userNum)
                        userNum += 1
                        satisfiedUsers = list(set(satisfiedUsers))  # remove duplicates
                    # if all the users satisfied reset it again
                    if len(satisfiedUsers) == self.usersPerGroup:
                        satisfiedUsers = []

            # ------------------------------- FAI -----------------------------------
            if self.groupModeling.lower() == "fai":
                rank = 0
                stack = []
                faiMatrix = relMatrix
                # recommendationsPerUser = math.ceil(self.numOfRecommendations / self.usersPerGroup)
                while rank <= self.numOfRecommendations:
                    if math.floor(rank / self.usersPerGroup) == 0:
                        availableIds = list(set(ids) - set(stack))
                        bestRel = -math.inf
                        for id in availableIds:
                            column = pd.to_numeric(faiMatrix[id])
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
                            column = pd.to_numeric(faiMatrix[id])
                            bestFilm = column.idxmax()
                            recommendations.append(bestFilm)
                            rank += 1
                    else:
                        for id in reversed(stack):
                            # print('id reversed', id)
                            column = pd.to_numeric(faiMatrix[id])
                            bestFilm = column.idxmax()
                            recommendations.append(bestFilm)
                            rank += 1
                    faiMatrix = faiMatrix.drop([bestFilm])

            # ------------------------------- XPO --------------------------------
            if self.groupModeling.lower() == "xpo":
                # get the N - level Pareto
                usersTopN = list()
                for id in ids:
                    sorted_ = relMatrix.sort_values(by=[id], ascending=False)
                    userTopN = list(sorted_.head(self.numOfRecommendations).index)
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
            if self.groupModeling.lower() == "arm":
                # lastRelFilm = [0] * self.usersPerGroup
                weights = [1] * self.usersPerGroup
                ourMatrix = relMatrix
                # work only with the top N individual recommendations
                individualRecom = []
                numOfIndRec = int(self.numOfRecommendations)
                for i in range(len(ids)):
                    ourMatrix = ourMatrix.sort_values(by=[ids[i]], ascending=False, inplace=False)
                    userRelItems = list(ourMatrix.index.values)[:numOfIndRec]
                    individualRecom.extend(userRelItems)
                individualRecom = list(set(individualRecom))
                ourMatrix = ourMatrix.loc[individualRecom]
                # balance the relevances
                min_ = ourMatrix.min(axis=0).values
                min_abs = np.absolute(min_)
                ourMatrix = ourMatrix.sub(min_, axis=1)
                ourMatrix = ourMatrix.add(min_abs, axis=1)
                #ourMatrix = ourMatrix.pow(1)  # increase the difference across members
                sum_ = ourMatrix.sum(axis=0).values  # / len(ourMatrix.index)  # / by the num of items so that all will have a relevance similar to 1
                #max_ = ourMatrix.max(axis=0).values
                ourMatrix = ourMatrix.div(sum_, axis=1)
                # minOfmin = min(ourMatrix.min(axis=0).values)
                # ourMatrix = ourMatrix.div(minOfmin, axis=1)
                #refMatrix = ourMatrix
                # ourMatrix = ourMatrix.pow(2)
                # newMin = ourMatrix.min(axis=0).values
                # newMin = newMin.min()
                for rank in range(self.numOfRecommendations):
                    weights = np.array(weights)
                    mean = sum(weights) / len(weights)
                    weights = mean / weights  # users with lower values will tend to radicalize more while user with higher will tend to accept more films
                    # weights = [1.2 if x>=1.2 else x for x in weights]  # necessary apply the root otherwise the
                    # print(weights)
                    # print('weights to power', weights)
                    # prevSum = refMatrix.sum(axis=0).values
                    # ourMatrix = ourMatrix.mul(weights, axis=1)
                    # ourMatrix = ourMatrix.mul(prevSum, axis=1)
                    ourMatrix = ourMatrix.pow(weights, axis=1)
                    # keepSum = ourMatrix.sum(axis=0).values / prevSum  # new sum / by old sum - > si abans la suma feia 0,8 que ara ho continui fent
                    ourMatrix["score"] = ourMatrix.sum(axis=1)
                    ourMatrix = ourMatrix.sort_values(by=['score'], ascending=False, inplace=False)
                    # print(ourMatrix)
                    recommendation = ourMatrix.index[0]
                    ourMatrix = ourMatrix.drop(columns=['score'])
                    weights = ourMatrix.iloc[0].values
                    # print('weights all different to 0', weights)
                    recommendations.append(recommendation)
                    ourMatrix = ourMatrix.drop([recommendation])
                    #refMatrix = refMatrix.drop([recommendation])

            # ------------------------------- OUR --------------------------------
            if self.groupModeling.lower() == "arm2":
                # lastRelFilm = [0] * self.usersPerGroup
                weights = [1] * self.usersPerGroup
                ourMatrix = relMatrix
                # work only with the top N individual recommendations
                individualRecom = []
                numOfIndRec = int(self.numOfRecommendations)
                for i in range(len(ids)):
                    ourMatrix = ourMatrix.sort_values(by=[ids[i]], ascending=False, inplace=False)
                    userRelItems = list(ourMatrix.index.values)[:numOfIndRec]
                    individualRecom.extend(userRelItems)
                individualRecom = list(set(individualRecom))
                ourMatrix = ourMatrix.loc[individualRecom]
                # balance the relevances
                min_ = ourMatrix.min(axis=0).values
                min_abs = np.absolute(min_)
                ourMatrix = ourMatrix.sub(min_, axis=1)
                ourMatrix = ourMatrix.add(min_abs, axis=1)
                #ourMatrix = ourMatrix.pow(1)  # increase the difference across members
                sum_ = ourMatrix.sum(axis=0).values  # / len(ourMatrix.index)  # / by the num of items so that all will have a relevance similar to 1
                #max_ = ourMatrix.max(axis=0).values
                ourMatrix = ourMatrix.div(sum_, axis=1)
                #to make sure the minimum value is at least 1
                #refMatrix = ourMatrix
                # ourMatrix = ourMatrix.pow(2)
                # newMin = ourMatrix.min(axis=0).values
                # newMin = newMin.min()
                for rank in range(self.numOfRecommendations):
                    weights = np.array(weights)
                    mean = sum(weights) / len(weights)
                    weights = mean / weights  # users with lower values will tend to radicalize more while user with higher will tend to accept more films
                    weights = [x**(2) for x in weights]  # necessary apply the root otherwise th e
                    # print('weights to power', weights)
                    # prevSum = refMatrix.sum(axis=0).values
                    # ourMatrix = ourMatrix.mul(weights, axis=1)
                    # ourMatrix = ourMatrix.mul(prevSum, axis=1)
                    ourMatrix = ourMatrix.mul(weights, axis=1)
                    # keepSum = ourMatrix.sum(axis=0).values / prevSum  # new sum / by old sum - > si abans la suma feia 0,8 que ara ho continui fent
                    ourMatrix["score"] = ourMatrix.sum(axis=1)
                    ourMatrix = ourMatrix.sort_values(by=['score'], ascending=False, inplace=False)
                    # print(ourMatrix)
                    recommendation = ourMatrix.index[0]
                    ourMatrix = ourMatrix.drop(columns=['score'])
                    weights = ourMatrix.iloc[0].values
                    # print('weights all different to 0', weights)
                    recommendations.append(recommendation)
                    ourMatrix = ourMatrix.drop([recommendation])
                    #refMatrix = refMatrix.drop([recommendation])


            # ------------------------------- REPUTATION --------------------------------
            if self.groupModeling.lower() == "reputation":
                rank = 0
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
                    reputationMatrix = reputationMatrix.drop([recommendation])
                    rank += 1

            recommendations_dict = {
                "recommendations": recommendations
            }
            allRecommnedations.append(recommendations_dict)
            # print(allRecommnedations[0]["recommendations"][0])
            # print(type(allRecommnedations[0]["recommendations"][0]))

        with open(self.path, 'w') as fout:
            json.dump(allRecommnedations, fout)
        bar.finish()
        elapsed_time = time.time() - start_time
        print("elapsed time", time.strftime("%H:%M:%S", time.gmtime(elapsed_time)), '\n')
        # Print the the Top_GN recommendations with each user relevances for the last group
        # print('The top recommendations for the last group are: \n')
        # top_g = relMatrix.reindex(recommendations)
        # print(top_g)
        return allRecommnedations

import pandas as pd
import numpy as np
import math


class reputation:

    def greedylm(self, lmMatrix, utility, k):
        # userUtility = ((utility[i] * k * relMax) + userRel) / ((k + 1) * relMax)
        relMax = lmMatrix.max().to_frame().T.values
        numerator = relMax * k
        denominator = relMax * (k + 1)
        mulDone = utility.mul(numerator, axis=1)
        sumDone = mulDone.add(lmMatrix)
        utility = sumDone.div(denominator, axis=1)
        return utility  # list of the utility of each member in the group

    def reputationBased(self, lambdaReputation, usersRatedItems, reputation, p):
        # print(len(relMatrix.columns))
        i = 0
        oldReputation = reputation
        for userRatedItem in usersRatedItems:
            # print(userRatedItem)
            seenItems = userRatedItem.iloc[:, 0]
            groupRel = userRatedItem.iloc[:, 1:]
            tempOldReputation = oldReputation[:i] + oldReputation[i + 1:]
            _ = groupRel * tempOldReputation
            groupScores = _.to_numpy().sum(axis=0) / sum(tempOldReputation)
            difference = sum((ru - gu)**p for ru, gu in zip(seenItems, groupScores)) ** (1 / p)
            reputation[i] = 1 - (lambdaReputation / (max(seenItems) * len(seenItems))) * difference
            i += 1
        return reputation  # list of the utility of each member in the group

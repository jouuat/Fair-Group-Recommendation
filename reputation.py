import pandas as pd
import numpy as np
import math


class reputation:

    def greedylm(self, recommendation, relMatrix, utility, k):
        print(len(relMatrix.columns))
        for i in range(len(relMatrix.columns)):
            ids = relMatrix.columns
            # print('ids:', ids)
            userRel = 0
            relMax = relMatrix[ids[i]].max()
            # with the average technique, there's also the proportional technique
            # userRel = relMatrix[ids[i]].at[recommendation]
            # userRel = relMatrix.at[recommendation, ids[i]]
            userRel = relMatrix.loc[recommendation, ids[i]]
            # print('relevance for user', ids[i], 'and recommendation', recommendation, 'has relevance', userRel, 'and k', k)
            # print('-----utility-----', utility, 'len(relMatrix)', len(relMatrix.columns))
            userUtility = ((utility[i] * k * relMax) + userRel) / ((k + 1) * relMax)
            utility[i] = userUtility
        return utility  # list of the utility of each member in the group

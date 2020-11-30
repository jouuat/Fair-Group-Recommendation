# import tensorflow as tf
# import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from scipy.stats import pearsonr


class groupDetection:
    def __init__(self, config, ratings_pd):
        self.groupDetection = config.groupDetection
        self.usersPerGroup = config.usersPerGroup
        self.numOfGroups = config.numOfGroups
        self.numOfFilmsPerGroup = config.numOfFilmsPerGroup
        self.ratings_pd = ratings_pd

    def detect(self):
        unique_users = self.ratings_pd["user_id"].unique().tolist()
        dataset = self.ratings_pd
        group = 0
        groups = list()  # llista de tots els grups amb els seus ids
        while group < self.numOfGroups:
            X = list()  # ids d'un grup nomes
            # random case
            if self.groupDetection.lower() == "random":
                X = np.random.choice(unique_users, size=self.usersPerGroup, replace=False)
                groups.append(X)
                group += 1
                continue

            # similar and distinct
            refUserId = np.random.choice(unique_users, replace=False)
            unique_users.remove(refUserId)
            refUser = dataset[dataset['user_id'] == refUserId]
            refMovies = refUser['movie_title'].tolist()
            refRatings = refUser['user_rating'].tolist()
            reference = pd.DataFrame(list(zip(refMovies, refRatings)),
                                     columns=['movie_title', refUserId])
            reference = reference.set_index('movie_title')
            if len(reference.index) <= 25:  # force to have a minimum of 25 movies rated
                continue
            X.append(refUserId)
            i = 0
            removedUsers = list()
            while i < (self.usersPerGroup - 1):
                newGroupMember = np.random.choice(unique_users, replace=False)
                new_ratings = dataset[dataset['user_id'] == newGroupMember]
                movIndices = new_ratings['movie_title'].tolist()
                ratings = new_ratings['user_rating'].tolist()
                newMember = pd.DataFrame(list(zip(movIndices, ratings)),
                                         columns=['movie_title', newGroupMember])
                newMember = newMember.set_index('movie_title')
                data = reference.join(newMember, how='inner')
                if len(data.index) < 6:  # there must be at list 6 ratings to have a significant pearson correlation
                    continue
                corr, _ = pearsonr(data[data.columns[0]], data[data.columns[-1]])
                if corr > 0.3 and self.groupDetection.lower() == "similar":
                    # print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    X.append(newGroupMember)
                    unique_users.remove(newGroupMember)
                    removedUsers.append(newGroupMember)
                if corr < 0.1 and self.groupDetection.lower() == "distinct":
                    # print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    X.append(newGroupMember)
                    unique_users.remove(newGroupMember)
                    removedUsers.append(newGroupMember)
            groups.append(X)
            unique_users = unique_users + removedUsers
            group += 1
        return groups

    def groupInfo(self):
        print(self.numOfGroups, " groups were created with", self.usersPerGroup, self.groupDetection, "users each \n")

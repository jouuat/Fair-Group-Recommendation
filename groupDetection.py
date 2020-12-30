# import tensorflow as tf
# import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import math
import json
import random
import progressbar


class groupDetection:
    def __init__(self, config, ratings_pd, path):
        self.groupDetection = config.groupDetection
        self.usersPerGroup = config.usersPerGroup
        self.numOfGroups = config.numOfGroups
        self.ratings_pd = ratings_pd
        self.allUsers = config.allUsers
        self.path = path

    def detect(self):
        users = self.ratings_pd["user_id"].unique().tolist()
        removed = []
        dataset = self.ratings_pd
        group = 0
        groups = list()  # llista de tots els grups amb els seus ids
        tries = 0
        notAsReference = []
        numOfUsers = len(users)
        bar = progressbar.ProgressBar(maxval=numOfUsers, widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
        bar.start()
        while (len(users) >= self.usersPerGroup) or (group == self.numOfGroups):  # don't compute the last 50 since it may be possible that there isn't more similar groups
            if tries == 50:
                print("not capable to make more groups with the remaining users")
                print(len(users), "users without group in comparison with", len(removed), "num of users with group")
                break
            X = list()  # ids d'un grup nomes
            pearsons = list()
            toBeRemoved = list()
            refUserId = random.choice(users)
            refUser = dataset[dataset['user_id'] == refUserId]
            refMovies = refUser['movie_title'].tolist()
            refRatings = refUser['user_rating'].tolist()
            reference = pd.DataFrame(list(zip(refMovies, refRatings)),
                                     columns=['movie_title', refUserId])
            reference = reference.set_index('movie_title')
            if (len(reference.index) <= 15) and (refUserId in notAsReference):  # force to have a minimum of 25 movies rated
                tries += 1
                continue
            X.append(refUserId)
            toBeRemoved.append(refUserId)
            users.remove(refUserId)  # necessary otherwise there may be duplicates when computing the pearsons
            i = 0
            tries = 0
            while i < (self.usersPerGroup - 1):
                tries += 1
                if tries < 40:
                    newGroupMember = random.choice(users)
                elif tries == 150:
                    notAsReference.append(refUserId)
                    users.append(refUserId)
                    break
                else:
                    newGroupMember = random.choice(removed)
                new_ratings = dataset[dataset['user_id'] == newGroupMember]
                movIndices = new_ratings['movie_title'].tolist()
                ratings = new_ratings['user_rating'].tolist()
                newMember = pd.DataFrame(list(zip(movIndices, ratings)),
                                         columns=['movie_title', newGroupMember])
                newMember = newMember.set_index('movie_title')
                data = reference.join(newMember, how='inner')
                if len(data.index) < 4:  # there must be at list 6 ratings to have a significant pearson correlation
                    continue
                corr, _ = pearsonr(data[refUserId], data[newGroupMember])
                if corr > 0.3 and self.groupDetection.lower() == "similar":
                    # print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    X.append(newGroupMember)
                    pearsons.append(corr)
                    if tries < 40:
                        users.remove(newGroupMember)
                    else:
                        removed.remove(newGroupMember)
                    toBeRemoved.append(newGroupMember)
                    tries = 0
                if corr < 0.1 and self.groupDetection.lower() == "distinct":
                    # print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    X.append(newGroupMember)
                    pearsons.append(corr)
                    if tries < 40:
                        users.remove(newGroupMember)
                    else:
                        removed.remove(newGroupMember)
                    toBeRemoved.append(newGroupMember)
                    tries = 0
                if self.groupDetection.lower() == "random":
                    # print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    X.append(newGroupMember)
                    pearsons.append(corr)
                    if tries < 40:
                        users.remove(newGroupMember)
                    else:
                        removed.remove(newGroupMember)
                    toBeRemoved.append(newGroupMember)
                    tries = 0

            removed.extend(toBeRemoved)
            group_dict = {
                "members": X,
                "pearsons": pearsons
            }
            groups.append(group_dict)
            if not self.allUsers:
                group += 1
            bar.update(numOfUsers - len(users))
        bar.finish()
        with open(self.path, 'w') as fout:
            json.dump(groups, fout)
        return groups

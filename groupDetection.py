# import tensorflow as tf
# import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import prince
from sklearn.cluster import KMeans
from preprocess import preprocess
from scipy.stats import pearsonr


class groupDetection(preprocess):
    def __init__(self, config, ratings, movie):
        super().__init__(config, ratings, movie)
        self.groupDetection = config.groupDetection

    def detect(self):
        # through k-means clustering
        if self.groupDetection == 1:
            X = self.preMcaKmeans()
            mca = prince.MCA(
                n_components=8,
                n_iter=3,
                copy=True,
                check_input=True,
                engine='auto',
                random_state=42
            )
            print('fitting the MCA model')
            mca = mca.fit(X)
            # dimImportance = mca.explained_inertia_
            coordinates = mca.row_coordinates(X)
            print(coordinates)
            group = KMeans(n_clusters=100, random_state=170).fit_predict(coordinates)  # 1000 diferent users just in the dataset
            X['group'] = group
            # print(X)
            # X[X['user_id'] == 1]
            return X
        if self.groupDetection > 1:
            # X = self.usersXmovies()
            unique_users = list(np.unique(self.user_id))
            dataset = np.column_stack((self.user_id, self.movie_title, self.user_rating.astype(int)))
            dataset = pd.DataFrame(data=dataset, columns=['user_id', 'movie_title', 'user_rating'])
            # print('starting the preprocessing')
            group = 0
            groups = list()  # llista de tots els grups amb els seus ids
            while group < self.numOfGroups:
                X = list()  # ids d'un grup nomes
                # random case
                if self.groupDetection == 4:
                    X = np.random.choice(unique_users, size=self.usersPerGroup, replace=False)
                    groups.append(X)
                    # print(groups)
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
                    if corr > 0.3 and self.groupDetection == 2:
                        # print('Pearsons correlation: %.3f' % corr)
                        i += 1
                        X.append(newGroupMember)
                        unique_users.remove(newGroupMember)
                        removedUsers.append(newGroupMember)
                    if corr < 0.1 and self.groupDetection == 3:
                        # print('Pearsons correlation: %.3f' % corr)
                        i += 1
                        X.append(newGroupMember)
                        unique_users.remove(newGroupMember)
                        removedUsers.append(newGroupMember)
                groups.append(X)
                unique_users = unique_users + removedUsers
                group += 1
                # print(group, "groups created")
                print(groups)
            return groups
        else:
            print("no dataset selected")

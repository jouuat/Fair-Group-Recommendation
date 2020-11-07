import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import pprint
import prince
from sklearn.cluster import KMeans
import random
from scipy.stats import pearsonr


class preprocess:
    def __init__(self, config, ratings, movies):
        self.ratings = ratings.map(lambda x: {
            "bucketized_user_age": x["bucketized_user_age"],
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "timestamp": x["timestamp"],
            "user_gender": x["user_gender"],
            "user_occupation_text": x["user_occupation_text"],
            "user_rating": x["user_rating"],
            "user_zip_code": x["user_zip_code"]
        })
        self.movies = movies.map(lambda x: x["movie_title"])
        self.usersPerGroup = config.usersPerGroup
        self.numOfGroups = config.numOfGroups
        self.groupDetection = config.groupDetection
        self.numOfFilmsPerGroup = config.numOfFilmsPerGroup
        self.movie_genres = 0
        self.candidates_title = np.concatenate(list(self.movies.batch(1000)))
        #self.bucketized_user_age = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["bucketized_user_age"])))
        self.movie_title = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["movie_title"])))
        self.user_id = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_id"])))  # .astype(int) perque funcioni be ajuntar els grups
        #self.timestamp = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["timestamp"])))
        #self.user_gender = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["user_gender"])))
        #self.user_occupation_text = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["user_occupation_text"])))
        self.user_rating = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_rating"])))
        #self.user_zip_code = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["user_zip_code"])))

    def preMcaKmeans(self):
        self.movie_genres = self.ratings.map(lambda x: x["movie_genres"])

        def selRand(t: np.ndarray):
            return np.random.choice(t, 1)
        self.movie_genres = self.movie_genres.map(lambda x: tf.numpy_function(func=selRand, inp=[x], Tout=tf.int64))
        self.movie_genres = np.concatenate(list(self.movie_genres))
        print(len(self.movie_genres))
        self.movie_title = self.ratings.map(lambda x: x["movie_title"])

        # error: 'ascii' codec can't decode byte 0xc3 in position 1: ordinal not in range(128)
        def decode(t: tf.Tensor):
            return t.numpy().decode('utf-8')
        self.movie_title = self.movie_title.map(lambda x: tf.py_function(func=decode, inp=[x], Tout=tf.string))
        self.movie_title = np.concatenate(list(self.movie_title.batch(1000)))
        # self.movie_title = np.char.decode(self.movie_title, encoding='utf_8')
        self.ratings = self.ratings.map(lambda x: {
            "bucketized_user_age": x["bucketized_user_age"],
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "timestamp": x["timestamp"],
            "user_gender": x["user_gender"],
            "user_occupation_text": x["user_occupation_text"],
            "user_rating": x["user_rating"],
            "user_zip_code": x["user_zip_code"]
        })

        self.bucketized_user_age = pd.qcut(self.bucketized_user_age, 6, labels=False, duplicates='drop')
        self.timestamp = pd.cut(self.timestamp, 10, labels=False)
        self.user_rating = pd.cut(self.user_rating, 5, labels=False)
        # self.movie_title = np.char.decode(self.movie_title)
        # self.user_zip_code = pd.cut(self.user_zip_code, 10, labels=False) es veu que tambe hi han postal codes amb lletres
        X = np.column_stack((self.user_id.astype(int), self.bucketized_user_age.astype(int), self.movie_genres, self.timestamp.astype(int),
                             self.user_gender, self.user_occupation_text, self.user_rating.astype(int), self.user_zip_code))
        X = pd.DataFrame(data=X, columns=['user_id', 'bucketized_user_age', 'movie_genres', 'timestamp', 'user_gender', 'user_occupation_text', 'user_rating', 'user_zip_code'])
        # X = X['movie_title'].str.decode(encoding='ASCII')
        # X = X.set_index('user_id')
        return X

    def usersXmovies(self):
        unique_users = list(np.unique(self.user_id))
        dataset = np.column_stack((self.user_id, self.movie_title, self.user_rating.astype(int)))
        dataset = pd.DataFrame(data=dataset, columns=['user_id', 'movie_title', 'user_rating'])
        print('starting the preprocessing')
        X = list()
        group = 0
        while group < self.numOfGroups:
            initial_user = np.random.choice(unique_users, replace=False)
            unique_users.remove(initial_user)
            initial_ratings = dataset[dataset['user_id'] == initial_user]  # & (df['nationality'] == "USA")
            movIndices = initial_ratings['movie_title'].tolist()
            ratings = initial_ratings['user_rating'].tolist()
            x = pd.DataFrame(list(zip(movIndices, ratings)),
                             columns=['movie_title', initial_user])
            x = x.set_index('movie_title')
            if len(x.index) <= 10 * self.usersPerGroup:
                continue
            X.append(x)
            i = 0
            tries = 0
            removedUsers = list()
            while i < (self.usersPerGroup - 1):
                if tries > 100:
                    del X[-1]
                    group -= 1
                    break
                # for i in range(self.usersPerGroup - 1):
                newGroupMember = np.random.choice(unique_users, replace=False)
                new_ratings = dataset[dataset['user_id'] == newGroupMember]  # & (df['nationality'] == "USA")
                movIndices = new_ratings['movie_title'].tolist()
                ratings = new_ratings['user_rating'].tolist()
                right = pd.DataFrame(list(zip(movIndices, ratings)),
                                     columns=['movie_title', newGroupMember])
                right = right.set_index('movie_title')
                '''
                left = X[group]
                data = left.join(right, how='outer')
                nanMovies = data.index[data.columns[-1] == nan].tolist()
                for j in range(len(nanMovies)):
                    predictedRating = predictRating(nanMovies[j], newGroupMember)
                    data.set_value(j, newGroupMember, predictedRating)
                '''
                # data = pd.merge(X[group], right, how='inner')
                data = X[group].join(right, how='inner')
                # data = pd.concat([X[group], right], axis=1, join="inner", sort=False)  # default join outer | Shape of passed values is (256, 2), indices imply (254, 2)
                if len(data.index) <= 0.5 * len(X[group]) or len(data.index) < 6:
                    tries += 1
                    #print("try:", tries)
                    continue
                corr, _ = pearsonr(data[data.columns[0]], data[data.columns[-1]])
                if corr > 0.3 and self.groupDetection == 2:
                    #print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    # print(data)
                    X[group] = data
                    unique_users.remove(newGroupMember)
                    removedUsers.append(newGroupMember)
                if corr < 0.1 and self.groupDetection == 3:
                    #print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    print(data)
                    X[group] = data
                    unique_users.remove(newGroupMember)
                    removedUsers.append(newGroupMember)
                if self.groupDetection == 4:
                    #print('Pearsons correlation: %.3f' % corr)
                    i += 1
                    # print(data)
                    X[group] = data
                    unique_users.remove(newGroupMember)
                    removedUsers.append(newGroupMember)
            unique_users.append(initial_user)
            unique_users = unique_users + removedUsers
            if group >= 1:
                print(X[group])

            group += 1
            print(group, "groups created")
        return X

        # print(X[1])

    def preRecommender(self):
        ratings = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
        })

        tf.random.set_seed(42)
        shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

        train = shuffled.take(80_000)
        test = shuffled.skip(80_000).take(20_000)
        return train, test

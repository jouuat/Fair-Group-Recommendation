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
        self.dataset = config.dataset
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
        # self.bucketized_user_age = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["bucketized_user_age"])))
        self.movie_title = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["movie_title"])))
        self.user_id = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_id"])))  # .astype(int) perque funcioni be ajuntar els grups
        # self.timestamp = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["timestamp"])))
        # self.user_gender = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["user_gender"])))
        # self.user_occupation_text = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["user_occupation_text"])))
        self.user_rating = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_rating"])))
        # self.user_zip_code = np.concatenate(list(self.ratings.batch(1).map(lambda x: x["user_zip_code"])))

    def dataInfo(self):
        numOfMovies = len(np.unique(self.movie_title).tolist())
        numOfusers = len(np.unique(self.user_id).tolist())
        numOfRatings = len(self.user_rating)
        print("The dataset", self.dataset, "contains", numOfRatings, "ratings of", numOfusers, "users and ", numOfMovies, "movies \n")

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

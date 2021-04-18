
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf
import json
import pprint

class dataset:

    def __init__(self, config):
        self.dataset = config.dataset
        self.ratings_pd = 0
        self.ratings = 0
        self.candidates = 0
        self.movie_title = 0
        self.user_id = 0
        self.user_rating = 0

    def getData(self):
        if self.dataset.lower() == "movielens100k":
            # {'bucketized_user_age': 45.0, 'movie_genres': array([7]), 'movie_id': b'357', 'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'raw_user_age': 46.0, 'timestamp': 879024327, 'user_gender': True, 'user_id': b'138', 'user_occupation_label': 4, 'user_occupation_text': b'doctor', 'user_rating': 4.0, 'user_zip_code': b'53211'}
            self.ratings = tfds.load("movielens/100k-ratings", split="train")
            # {'movie_genres': array([4]), 'movie_id': b'1681', 'movie_title': b'You So Crazy (1994)'}
            self.candidates = tfds.load("movielens/100k-movies", split="train")
        if self.dataset.lower() == "movielens1m":
            # {'bucketized_user_age': 45.0, 'movie_genres': array([7]), 'movie_id': b'357', 'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'raw_user_age': 46.0, 'timestamp': 879024327, 'user_gender': True, 'user_id': b'138', 'user_occupation_label': 4, 'user_occupation_text': b'doctor', 'user_rating': 4.0, 'user_zip_code': b'53211'}
            self.ratings = tfds.load("movielens/1m-ratings", split="train")
            # {'movie_genres': array([4]), 'movie_id': b'1681', 'movie_title': b'You So Crazy (1994)'}
            self.candidates = tfds.load("movielens/1m-movies", split="train")
        if self.dataset.lower() == "movielens1m" or self.dataset.lower() == "movielens100k":
            self.transform_tfds_movie_lens()
        if self.dataset.lower() == "amazondigitalsoftware":
            self.ratings = tfds.load("amazon_us_reviews/Digital_Software_v1_00", split="train")
            self.transform_amazon_dig_soft()
        if self.dataset.lower() == "amazongrocery5":
            self.ratings = pd.read_json('grocery_and_gourmet_food_5.json', lines=True)
            self.transform_amazon_grocery_5()
        if self.dataset.lower() == "amazoninst5":
            self.ratings = pd.read_json('Musical_Instruments_5.json', lines=True)
            self.transform_amazon_inst_5()
        self.dataInfo()
        return self.ratings, self.candidates, self.ratings_pd

    def transform_amazon_inst_5(self):
        self.ratings = self.ratings[["reviewerID", "asin", "overall"]]
        self.ratings = self.ratings.rename(columns={"asin": "movie_title", "reviewerID": "user_id", "overall": "user_rating"})
        self.ratings = self.ratings.sample(frac=1).reset_index(drop=True)
        self.ratings_pd = self.ratings
        self.ratings['user_rating']= self.ratings['user_rating'].astype(float)
        self.ratings = tf.data.Dataset.from_tensor_slices(dict(self.ratings_pd))
        # self.candidates = self.ratings_pd["movie_title"]
        # self.candidates = tf.data.Dataset.from_tensor_slices(dict(self.candidates))
        self.candidates = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"]
        })
        #for element in self.ratings.take(2).as_numpy_iterator():
        #    pprint.pprint(element)

    def transform_amazon_grocery_5(self):
        self.ratings = self.ratings[["reviewerID", "asin", "overall"]]
        self.ratings = self.ratings.rename(columns={"asin": "movie_title", "reviewerID": "user_id", "overall": "user_rating"})
        self.ratings = self.ratings.sample(frac=1).reset_index(drop=True)
        self.ratings_pd = self.ratings
        self.ratings['user_rating']= self.ratings['user_rating'].astype(float)
        self.ratings = tf.data.Dataset.from_tensor_slices(dict(self.ratings_pd))
        # self.candidates = self.ratings_pd["movie_title"]
        # self.candidates = tf.data.Dataset.from_tensor_slices(dict(self.candidates))
        self.candidates = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"]
        })
        #for element in self.ratings.take(2).as_numpy_iterator():
        #    pprint.pprint(element)


    def transform_amazon_dig_soft(self):
        # self.movies = self.movies.map(lambda x: x["movie_title"])
        self.ratings = self.ratings.map(lambda x: {
            "movie_title": x["data"]["product_id"],
            "user_id": x["data"]["customer_id"],
            "user_rating": x["data"]["star_rating"]
        })  # a necessary step to be able to vatch more than 1 element'''
        self.candidates = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"]
        })
        tf.random.set_seed(42)
        self.ratings = self.ratings.shuffle(100000, seed=42, reshuffle_each_iteration=False)  # shuffle ratings
        self.movie_title = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["movie_title"])))
        decoder = np.vectorize(lambda x: x.decode('UTF-8'))  # necessary decode before transforming to a string
        self.movie_title = decoder(self.movie_title)
        self.user_id = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_id"]))).astype('U13')
        self.user_rating = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_rating"])))
        ratings_np = np.column_stack((self.user_id, self.movie_title, self.user_rating.astype(int)))
        self.ratings_pd = pd.DataFrame(data=ratings_np, columns=['user_id', 'movie_title', 'user_rating'])  # no index specified for the moment
        self.ratings_pd.user_rating = self.ratings_pd.user_rating.astype(int)
        # self.ratings_pd.movie_title = self.ratings_pd.movie_title.apply(str)


    def transform_tfds_movie_lens(self):
        # self.movies = self.movies.map(lambda x: x["movie_title"])
        self.ratings = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"]
        })  # a necessary step to be able to vatch more than 1 element'''
        #for element in self.ratings.take(2).as_numpy_iterator():
        #    pprint.pprint(element)
        tf.random.set_seed(42)
        self.ratings = self.ratings.shuffle(100000, seed=42, reshuffle_each_iteration=False)  # shuffle ratings
        self.movie_title = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["movie_title"])))
        decoder = np.vectorize(lambda x: x.decode('UTF-8'))  # necessary decode before transforming to a string
        self.movie_title = decoder(self.movie_title)
        self.user_id = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_id"]))).astype('U13')
        self.user_rating = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_rating"])))
        ratings_np = np.column_stack((self.user_id, self.movie_title, self.user_rating.astype(int)))
        self.ratings_pd = pd.DataFrame(data=ratings_np, columns=['user_id', 'movie_title', 'user_rating'])  # no index specified for the moment
        self.ratings_pd.user_rating = self.ratings_pd.user_rating.astype(int)
        # self.ratings_pd.movie_title = self.ratings_pd.movie_title.apply(str)


    def dataInfo(self):
        numOfMovies = len(np.unique(self.ratings_pd["movie_title"]).tolist())
        numOfusers = len(np.unique(self.ratings_pd["user_id"]).tolist())
        numOfRatings = len(self.ratings_pd)
        print("The dataset", self.dataset, "contains", numOfRatings, "ratings of", numOfusers, "users and ", numOfMovies, "movies \n")

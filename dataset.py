
import numpy as np
import pandas as pd
import tensorflow_datasets as tfds
import tensorflow as tf


class dataset:

    def __init__(self, config):
        self.dataset = config.dataset
        self.ratings_pd = 0
        self.ratings = 0
        self.movies = 0
        self.movie_title = 0
        self.user_id = 0
        self.user_rating = 0

    def getData(self):
        if self.dataset.lower() == "tfds_movie_lens_100k":
            # {'bucketized_user_age': 45.0, 'movie_genres': array([7]), 'movie_id': b'357', 'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'raw_user_age': 46.0, 'timestamp': 879024327, 'user_gender': True, 'user_id': b'138', 'user_occupation_label': 4, 'user_occupation_text': b'doctor', 'user_rating': 4.0, 'user_zip_code': b'53211'}
            self.ratings = tfds.load("movie_lens/100k-ratings", split="train")
            # {'movie_genres': array([4]), 'movie_id': b'1681', 'movie_title': b'You So Crazy (1994)'}
            self.movies = tfds.load("movie_lens/100k-movies", split="train")
            self.transorm_tfds_movie_lens()
            self.dataInfo()
            return self.ratings, self.movies, self.ratings_pd
        else:
            print("no dataset selected")

    def transorm_tfds_movie_lens(self):
        self.movies = self.movies.map(lambda x: x["movie_title"])
        self.ratings = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"]
        })  # a necessary step to be able to vatch more than 1 element
        # tf.random.set_seed(42)
        self.ratings = self.ratings.shuffle(100000, seed=42, reshuffle_each_iteration=False)  # shuffle ratings
        self.movie_title = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["movie_title"])))
        self.user_id = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_id"])))
        self.user_rating = np.concatenate(list(self.ratings.batch(1000).map(lambda x: x["user_rating"])))
        ratings_np = np.column_stack((self.user_id, self.movie_title, self.user_rating.astype(int)))
        self.ratings_pd = pd.DataFrame(data=ratings_np, columns=['user_id', 'movie_title', 'user_rating'])  # no index specified for the moment
        self.ratings = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
        })

    def dataInfo(self):
        numOfMovies = len(np.unique(self.movie_title).tolist())
        numOfusers = len(np.unique(self.user_id).tolist())
        numOfRatings = len(self.user_rating)
        print("The dataset", self.dataset, "contains", numOfRatings, "ratings of", numOfusers, "users and ", numOfMovies, "movies \n")

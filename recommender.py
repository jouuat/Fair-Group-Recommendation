import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from annoy import AnnoyIndex
import pandas as pd

import tensorflow_recommenders as tfrs
from preprocess import preprocess


class recommendationModel(preprocess):
    def __init__(self, config, ratings, movies):
        super().__init__(config, ratings, movies)
        self.embedding_dimension = config.embedding_dimension
        self.model = 0

    def train(self):
        train, test = self.preRecommender()
        unique_movie_titles = np.unique(self.candidates_title).tolist()
        unique_user_ids = np.unique(self.user_id)
        # unique_movie_titles[:10]
        model = recommenderModel(self.embedding_dimension, unique_user_ids, unique_movie_titles, self.movies)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))

        cached_train = train.shuffle(100000).batch(8192).cache()
        cached_test = test.batch(4096).cache()

        model.fit(cached_train, epochs=3)
        model.evaluate(cached_test, return_dict=True)
        self.model = model
        self.predict(unique_user_ids)

    def predict(self, users):
        '''index = AnnoyIndex(self.embedding_dimension, "dot")

        movie_embeddings = self.movies.enumerate().map(lambda idx, title: (idx, title, self.model.movie_model(title)))  #  (7,8) -enumerate()->(0,array([7,8], dtype=int32))

        movie_id_to_title = dict((idx, title) for idx, title, _ in movie_embeddings.as_numpy_iterator())

        # We unbatch the dataset because Annoy accepts only scalar (id, embedding) pairs.
        for movie_id, _, movie_embedding in movie_embeddings.as_numpy_iterator():
            index.add_item(movie_id, movie_embedding)

        # Build a 10-tree ANN index.
        index.build(10)'''

        '''for row in users.batch(1).take(3):
            query_embedding = self.model.user_model(row["user_id"])[0]
            print(row["user_id"])
            # candidates = index.get_nns_by_vector(query_embedding, 3)
            query_embedding = tf.reshape(query_embedding, [1, 32])
            puntuations = self.model.puntuations(query_embedding)
            print(puntuations)
            # print(f"Candidates: {[movie_id_to_title[x] for x in candidates]}.")'''
        users = users.tolist()
        usertensor = tf.convert_to_tensor(users[0], dtype=tf.string)
        usertensor = tf.reshape(usertensor, [1, ])
        query_embedding = self.model.user_model(usertensor)[0]
        query_embedding = tf.reshape(query_embedding, [1, 32])
        puntuations = self.model.puntuations(query_embedding)
        scores = puntuations[0].numpy()
        print(scores)
        movies = puntuations[1].numpy()
        print(movies)
        numpy_data = np.concatenate((movies.T, scores.T), axis=1)
        # print(numpy_data)
        X = pd.DataFrame(data=numpy_data, columns=["movies", users[0]])
        #X = pd.DataFrame(list(zip(movies, scores)), columns=['movies', users[0]])
        X = X.set_index("movies")
        print(X)
        users.remove(users[0])

        for user in users:
            X.drop_duplicates(inplace=True)
            usertensor = tf.convert_to_tensor(user, dtype=tf.string)
            usertensor = tf.reshape(usertensor, [1, ])
            query_embedding = self.model.user_model(usertensor)[0]
            query_embedding = tf.reshape(query_embedding, [1, 32])
            puntuations = self.model.puntuations(query_embedding)
            scores = puntuations[0].numpy()
            movies = puntuations[1].numpy()
            numpy_data = np.concatenate((movies.T, scores.T), axis=1)
            x = pd.DataFrame(data=numpy_data, columns=["movies", user])
            x = x.set_index("movies")
            x.drop_duplicates(inplace=True)
            print(x)
            X = X.join(x, how='inner')
            #X = pd.concat([X, x], axis=1, join='inner')
            print(X)
            # print(f"Candidates: {[movie_id_to_title[x] for x in candidates]}.")'''
        print(X)


class userModel(tf.keras.Model):
    def __init__(self, embedding_dimension, unique_user_ids):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])

    def call(self, ids):
        return self.user_embedding(ids)


class movieModel(tf.keras.Model):

    def __init__(self, embedding_dimension, unique_movie_titles):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

    def call(self, titles):
        return self.title_embedding(titles)


class recommenderModel(tfrs.Model):

    def __init__(self, embedding_dimension, unique_user_ids, unique_movie_titles, movies):
        super().__init__()
        self.movie_model: tf.keras.Model = movieModel(embedding_dimension, unique_movie_titles)
        self.user_model: tf.keras.Model = userModel(embedding_dimension, unique_user_ids)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.movie_model)

            ))
        #tensorCandidates = tf.convert_to_tensor(unique, dtype=tf.string)
        #candidates = list(map(lambda x: (tf.convert_to_tensor(x, dtype=tf.string), self.movie_model(tf.convert_to_tensor(x, dtype=tf.string))), unique_movie_titles))
        candidates = tf.data.Dataset.from_tensor_slices(unique_movie_titles)
        #tensorCandidates = tf.convert_to_tensor(candidates, dtype=tf.string)
        self.puntuations = tfrs.layers.corpus.DatasetIndexedTopK(
            candidates=candidates.batch(128).map(lambda title: (title, self.movie_model(title))),
            k=len(unique_movie_titles)
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        positive_movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, positive_movie_embeddings)
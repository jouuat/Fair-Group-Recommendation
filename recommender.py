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


class recommendationModel:
    def __init__(self, config, ratings, movies, ratings_pd):
        self.embedding_dimension = config.embedding_dimension
        self.model = 0
        self.ratings = ratings
        self.ratings_pd = ratings_pd
        self.movies = movies

    def train(self):
        unique_movie_titles = self.ratings_pd["movie_title"].unique()
        unique_user_ids = self.ratings_pd["user_id"].unique()
        model = recommenderModel(self.embedding_dimension, unique_user_ids, unique_movie_titles, self.movies)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        # split in train and test
        sizeDataset = len(self.ratings_pd["user_rating"].tolist())
        train = self.ratings.take(int(sizeDataset * 0.8))
        test = self.ratings.skip(int(sizeDataset * 0.8)).take(int(sizeDataset * 0.2))
        cached_train = train.shuffle(100000).batch(8192).cache()
        cached_test = test.batch(4096).cache()
        # define the callbacks
        checkpoint_filepath = '/tmp/checkpoint'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='loss', mode='min')

        stp_callback = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=5,
        )
        class CustomPrintingCallback(tf.keras.callbacks.Callback):

            def on_epoch_end(self, epoch, logs=None):
                print('Average loss for epoch {} is {:7.2f}.\n'.format(epoch, logs['loss']))
        # fit and evaluate the model
        model.fit(cached_train, epochs=1, verbose=0, callbacks=[stp_callback, cp_callback, CustomPrintingCallback()])
        model.evaluate(cached_test, return_dict=True, verbose=0)
        self.model = model
        return model

    def predict(users, model):
        usertensor = tf.convert_to_tensor(users[0], dtype=tf.string)
        usertensor = tf.reshape(usertensor, [1, ])
        query_embedding = model.user_model(usertensor)[0]
        query_embedding = tf.reshape(query_embedding, [1, 32])
        puntuations = model.puntuations(query_embedding)
        scores = puntuations[0].numpy()
        movies = puntuations[1].numpy()
        numpy_data = np.concatenate((movies.T, scores.T), axis=1)
        X = pd.DataFrame(data=numpy_data, columns=["movies", users[0]])
        X = X.set_index("movies")

        for user in users[1:]:
            X.drop_duplicates(inplace=True)
            usertensor = tf.convert_to_tensor(user, dtype=tf.string)
            usertensor = tf.reshape(usertensor, [1, ])
            query_embedding = model.user_model(usertensor)[0]
            query_embedding = tf.reshape(query_embedding, [1, 32])
            puntuations = model.puntuations(query_embedding)
            scores = puntuations[0].numpy()
            movies = puntuations[1].numpy()
            numpy_data = np.concatenate((movies.T, scores.T), axis=1)
            x = pd.DataFrame(data=numpy_data, columns=["movies", user])
            x = x.set_index("movies")
            x.drop_duplicates(inplace=True)
            X = X.join(x, how='inner')
        return X


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

        max_tokens = 10000

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
        candidates = tf.data.Dataset.from_tensor_slices(unique_movie_titles)
        self.puntuations = tfrs.layers.corpus.DatasetIndexedTopK(
            candidates=candidates.batch(128).map(lambda title: (title, self.movie_model(title))),
            k=len(unique_movie_titles)
        )

    def compute_loss(self, features: Dict[Text, tf.Tensor], training=False) -> tf.Tensor:
        user_embeddings = self.user_model(features["user_id"])
        positive_movie_embeddings = self.movie_model(features["movie_title"])
        return self.task(user_embeddings, positive_movie_embeddings)

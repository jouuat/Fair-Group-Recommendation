import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from metrics import metrics
import progressbar


import tensorflow_recommenders as tfrs


class recommendationModel:
    def __init__(self, config, ratings, movies, ratings_pd):
        self.embedding_dimension = config.embedding_dimension
        self.model = 0
        self.ratings = ratings
        self.ratings_pd = ratings_pd
        self.movies = movies
        self.metricsForRecommender = config.metricsForRecommender
        self.metric = config.metric
        self.numOfRecommendations = config.numOfRecommendations
        self.layers = list(config.layers)

    def train(self):
        # split in train and test
        ratings = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
        })
        movies = self.movies.map(lambda x: x["movie_title"])

        sizeDataset = len(self.ratings_pd["user_rating"].tolist())
        unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))  # self.ratings_pd["movie_title"].unique()
        unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
            lambda x: x["user_id"]))))  # self.ratings_pd["user_id"].unique()

        train = ratings.take(int(sizeDataset * 0.8))
        test = ratings.skip(int(sizeDataset * 0.8)).take(int(sizeDataset * 0.2))
        cached_train = train.shuffle(100000).batch(2048)
        cached_test = test.batch(2048).cache()

        # define the callbacks
        checkpoint_filepath = '/tmp/checkpoint'
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, save_weights_only=True, monitor='accuracy', mode='max')

        stp_callback = tf.keras.callbacks.EarlyStopping(
            monitor='factorized_top_k/top_100_categorical_accuracy', patience=5,
        )
        class CustomPrintingCallback(tf.keras.callbacks.Callback):

            def on_epoch_end(self, epoch, logs=None):
                print('The proportion of true candidates in Top-100 for epoch {} is {:7.2f}.\n'.format(epoch, logs['factorized_top_k/top_100_categorical_accuracy']))
        # fit and evaluate the model
        model = recommenderModel(self.layers, self.embedding_dimension, unique_user_ids, unique_movie_titles, movies)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        model.fit(
            cached_train,
            validation_data=cached_test,
            validation_freq=5,
            epochs=100,
            verbose=1,
            callbacks=[stp_callback, cp_callback]
        )
        # model.fit(cached_train, epochs=10, verbose=1, callbacks=[stp_callback, cp_callback])  # , CustomPrintingCallback()])  # ------------- HERE ------------
        # model.evaluate(cached_test, return_dict=True, verbose=1)
        self.model = model
        if self.metricsForRecommender:
            print("Computing metrics for the recommender")
            totalScores = []
            bar = progressbar.ProgressBar(maxval=len(unique_user_ids), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
            bar.start()
            i = 0
            for id in unique_user_ids:
                i += 1
                bar.update(i)
                users = []
                users.append(id)
                relMatrix = recommendationModel.predict(users, model)
                recommendations = list(relMatrix.index.values[:self.numOfRecommendations])
                test_pd = self.ratings_pd.tail(int(sizeDataset * 0.2))
                metrics_ = metrics(test_pd, users, recommendations)
                groupScores = metrics_.getScore(self.metric)
                totalScores.append(groupScores[0])
            totalScore = (sum(totalScores) / len(unique_user_ids))
            print(totalScores)
            print(self.metric, ':', totalScore)
        return model

    def predict(users, model):
        usertensor = tf.convert_to_tensor(users[0], dtype=tf.string)
        usertensor = tf.reshape(usertensor, [1, ])
        query_embedding = model.query_model(usertensor)[0]
        query_embedding = tf.reshape(query_embedding, [1, 32])
        puntuations = model.puntuations(query_embedding)
        scores = puntuations[0].numpy()
        movies = puntuations[1].numpy()
        numpy_data = np.concatenate((movies.T, scores.T), axis=1)
        X = pd.DataFrame(data=numpy_data, columns=["movies", users[0]])
        X = X.set_index("movies")
        if len(users) == 1:
            return X
        for user in users[1:]:
            X.drop_duplicates(inplace=True)
            usertensor = tf.convert_to_tensor(user, dtype=tf.string)
            usertensor = tf.reshape(usertensor, [1, ])
            query_embedding = model.query_model(usertensor)[0]
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


class queryModel(tf.keras.Model):
    def __init__(self, layer_sizes, embedding_dimension, unique_user_ids):
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = userModel(embedding_dimension, unique_user_ids)

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class movieModel(tf.keras.Model):
    def __init__(self, embedding_dimension, unique_movie_titles, movies):
        super().__init__()

        max_tokens = 10000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])

        self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, embedding_dimension, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, titles):
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)


class candidateModel(tf.keras.Model):
    def __init__(self, layer_sizes, embedding_dimension, unique_movie_titles, movies):
        super().__init__()

        self.embedding_model = movieModel(embedding_dimension, unique_movie_titles, movies)

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)
        return self.dense_layers(feature_embedding)


class recommenderModel(tfrs.Model):

    def __init__(self, layer_sizes, embedding_dimension, unique_user_ids, unique_movie_titles, movies):
        super().__init__()
        self.query_model = queryModel(layer_sizes, embedding_dimension, unique_user_ids)
        self.candidate_model = candidateModel(layer_sizes, embedding_dimension, unique_movie_titles, movies)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model),
            ),
        )
        candidates = tf.data.Dataset.from_tensor_slices(unique_movie_titles)
        self.puntuations = tfrs.layers.corpus.DatasetIndexedTopK(
            candidates=candidates.batch(128).map(lambda title: (title, self.candidate_model(title))),
            k=len(unique_movie_titles)
        )

    def compute_loss(self, features, training=False):
        query_embeddings = self.query_model(features["user_id"])
        movie_embeddings = self.candidate_model(features["movie_title"])
        return self.task(query_embeddings, movie_embeddings)

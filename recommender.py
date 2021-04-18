import os
import pprint
import tempfile
from typing import Dict, Text
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
from metrics import metrics
# import tensorflow_ranking as tfr
from os import path
import json
import progressbar

import tensorflow_recommenders as tfrs


class recommendationModel:
    def __init__(self, config, ratings, movies, ratings_pd, checkpoint_filepath):
        self.config = config
        self.embedding_dimension = config.embedding_dimension
        self.model = 0
        self.ratings = ratings
        self.ratings_pd = ratings_pd
        self.movies = movies
        self.metricsForRecommender = config.metricsForRecommender
        self.metric = config.metric
        self.numOfRecommendations = config.numOfRecommendations
        self.layers = list(config.layers)
        self.rankingWeight = config.rankingWeight
        self.retrievalWeight = config.retrievalWeight
        self.ratingWeight = config.ratingWeight
        self.epochs = config.epochs
        self.checkpoint_filepath = checkpoint_filepath
        self.dataset = config.dataset

    def train(self):
        # split in train and test
        ratings = self.ratings.map(lambda x: {
            "movie_title": x["movie_title"],
            "user_id": x["user_id"],
            "user_rating": x["user_rating"],
        })
        movies = self.movies.map(lambda x: x["movie_title"])
        sizeDataset = len(self.ratings_pd["user_rating"].tolist())
        train_pd = self.ratings_pd.head(int(sizeDataset * 0.8))
        test_pd = self.ratings_pd.tail(int(sizeDataset * 0.2))

        unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))  # self.ratings_pd["movie_title"].unique()
        unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1_000).map(
            lambda x: x["user_id"]))))  # self.ratings_pd["user_id"].unique()

        train = ratings.take(int(sizeDataset * 0.8))
        test = ratings.skip(int(sizeDataset * 0.8)).take(int(sizeDataset * 0.2))
        cached_train = train.shuffle(100000).batch(2048)
        cached_test = test.batch(2048).cache()

        # define the callbacks
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_filepath, save_weights_only=True, monitor='factorized_top_k', mode='max', save_best_only=True)
        stp_callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_root_mean_squared_error', patience=2, verbose=1, mode='min', restore_best_weights=True
        )

        class CustomPrintingCallback(tf.keras.callbacks.Callback):
            def on_epoch_end(self, epoch, logs=None):
                print('The Top K facorized of epoch {} is {:7.2f}.\n'.format(epoch, logs['factorized_top_k']))
        # fit and evaluate the model
        model = recommenderModel(self.rankingWeight, self.retrievalWeight, self.ratingWeight, self.layers,
                                 self.embedding_dimension, unique_user_ids, unique_movie_titles, movies, self.numOfRecommendations)
        model.compile(optimizer=tf.keras.optimizers.Adagrad(learning_rate=0.1))
        if tf.train.latest_checkpoint(self.checkpoint_filepath):
            print("Loading weights from {}".format(manager.latest_checkpoint))
            self.model = model.load_weights(self.checkpoint_filepath)
        else:
            print("Initializing from scratch.")
            model.fit(
                cached_train,
                validation_data=cached_test,
                validation_freq=2,
                epochs=self.epochs,
                verbose=1,
                # callbacks=[cp_callback]  # , stp_callback, CustomPrintingCallback()]
            )
            self.model = model

        # compute ndcg for the recommender results
        if self.metricsForRecommender:
            print("Computing metrics for the recommender")
            totalScores = []
            i = 0
            allRecommendations = []
            groups = []
            unique_user_ids = unique_user_ids.astype('U13')
            group_path = 'groups_' + str(self.dataset) + '_recommender.txt'
            recommendation_path = 'recommendations_' + str(self.dataset) + '_recommender.txt'
            if path.exists(recommendation_path) and path.exists(group_path):
                with open(group_path, "r") as read_file:
                    groups = json.load(read_file)
                with open(recommendation_path, "r") as read_file:
                    allRecommendations = json.load(read_file)
            else:
                print("obtaining the recommender recommendations")
                bar = progressbar.ProgressBar(maxval=len(unique_user_ids), widgets=[progressbar.Bar('=', '[', ']'), ' ', progressbar.Percentage()])
                bar.start()
                for id in unique_user_ids:
                    i += 1
                    users = [id]
                    relMatrix = recommendationModel.predict(users, model)
                    # Delete those films that has been already seen by the user
                    _ = train_pd[train_pd['user_id'] == id]
                    seenItems = _['movie_title']
                    relMatrix = relMatrix.drop(seenItems)
                    group_dict = {
                        "members": users
                    }
                    groups.append(group_dict)
                    recommendations = list(relMatrix.index.values[:self.numOfRecommendations])
                    recommendations_dict = {
                        "recommendations": recommendations
                    }
                    allRecommendations.append(recommendations_dict)
                    bar.update(i)
                with open(group_path, 'w') as fout:
                    json.dump(groups, fout)
                with open(recommendation_path, 'w') as fout:
                    json.dump(allRecommendations, fout)
                bar.finish()
            metrics_ = metrics(self.config, self.ratings_pd, groups, allRecommendations)
            score = metrics_.getScore()
            print(self.metric, ':', score)
        return model

    def predict(users, model):
        usertensor = tf.convert_to_tensor(users[0], dtype=tf.string)
        usertensor = tf.reshape(usertensor, [1, ])
        query_embedding = model.query_model(usertensor)[0]
        query_embedding = tf.reshape(query_embedding, [1, 32])
        puntuations = model.puntuations(query_embedding)
        scores = puntuations[0].numpy()
        movies = puntuations[1].numpy()
        decoder = np.vectorize(lambda x: x.decode('UTF-8'))  # necessary decode before transforming to a string
        movies = decoder(movies)
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
            decoder = np.vectorize(lambda x: x.decode('UTF-8'))  # necessary decode before transforming to a string
            movies = decoder(movies)
            numpy_data = np.concatenate((movies.T, scores.T), axis=1)
            x = pd.DataFrame(data=numpy_data, columns=["movies", user])
            x = x.set_index("movies")
            x.drop_duplicates(inplace=True)
            X = X.join(x, how='inner')
        X = X.astype(float)
        # print('duplicates', X[X.index.duplicated()])
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

    def __init__(self, ranking_weight, retrieval_weight, rating_weight, layer_sizes, embedding_dimension, unique_user_ids, unique_movie_titles, movies, numOfRecommendations):
        super().__init__()
        # user, items and ratings model
        # self.query_model = queryModel(layer_sizes, embedding_dimension, unique_user_ids)
        # self.candidate_model = candidateModel(layer_sizes, embedding_dimension, unique_movie_titles, movies)
        self.candidate_model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, embedding_dimension)
        ])
        self.query_model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, embedding_dimension)
        ])
        self.rating_model = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation="relu"),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(1),
        ])
        # the tasks -> losses and metrics
        # loss=tfr.keras.losses.get(tfr.losses.RankingLossKey.SOFTMAX_LOSS), --> ndcg = 0.01
        self.rating_task: tf.keras.layers.Layer = tfrs.tasks.Ranking(
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.RootMeanSquaredError()],
        )
        self.ranking_task = tfrs.tasks.Ranking()
        self.retrieval_task: tf.keras.layers.Layer = tfrs.tasks.Retrieval(
            # loss=tfr.keras.losses.get(tfr.losses.RankingLossKey.APPROX_NDCG_LOSS), don't work
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model)
            )
        )
        # prediction
        candidates_ = tf.data.Dataset.from_tensor_slices(unique_movie_titles)
        self.puntuations = tfrs.layers.corpus.DatasetIndexedTopK(
            candidates=candidates_.batch(128).map(lambda title: (title, self.candidate_model(title))),
            k=len(unique_movie_titles)
        )
        # loss weights
        self.rating_weight = rating_weight
        self.retrieval_weight = retrieval_weight
        self.ranking_weight = ranking_weight

    def call(self, features):
        user_embeddings = self.query_model(features["user_id"])
        item_embeddings = self.candidate_model(features["movie_title"])
        return (
            user_embeddings,
            item_embeddings,
            self.rating_model(
                tf.concat([user_embeddings, item_embeddings], axis=1)
            ),
        )

    def compute_loss(self, features, training=False):
        ratings = features.pop("user_rating")
        user_embeddings, item_embeddings, rating_predictions = self(features)
        rating_loss = self.rating_task(
            labels=ratings,
            predictions=rating_predictions,
        )
        ranking_loss = self.ranking_task(user_embeddings, item_embeddings)
        retrieval_loss = self.retrieval_task(user_embeddings, item_embeddings)

        return (self.rating_weight * rating_loss + self.ranking_weight * ranking_loss + self.retrieval_weight * retrieval_loss) # self.retrieval_weight * retrieval_loss

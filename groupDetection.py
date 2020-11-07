import tensorflow as tf
import tensorflow_datasets as tfds
import pandas as pd
import numpy as np
import prince
from sklearn.cluster import KMeans
from preprocess import preprocess


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
            dimImportance = mca.explained_inertia_
            coordinates = mca.row_coordinates(X)
            print(coordinates)
            group = KMeans(n_clusters=100, random_state=170).fit_predict(coordinates)  # 1000 diferent users just in the dataset
            X['group'] = group
            # print(X)
            #X[X['user_id'] == 1]
            return X
        if self.groupDetection > 1:
            X = self.usersXmovies()
            return X
        else:
            print("no dataset selected")

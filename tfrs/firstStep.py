import tensorflow as tf
import pprint

import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs

# Ratings data.
ratings = tfds.load("movie_lens/100k-ratings", split="train")
# Features of all the available movies.
movies = tfds.load("movie_lens/100k-movies", split="train")

for x in ratings.take(1).as_numpy_iterator():
    print(x)

# Use as_numpy_iterator to inspect the content of your dataset.
for x in movies.take(1).as_numpy_iterator():
    pprint.pprint(x)

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
})
movies = movies.map(lambda x: x["movie_title"])

print('-------------------')

tf.print(movies)

print('-------------------')

# pprint.pprint(list(movies.as_numpy_iterator()))

# When using only user ids and movie titles our simple two-tower model is very similar to a typical matrix factorization model. To build it, we're going to need the following:
#     A user tower that turns user ids into user embeddings (high-dimensional vector representations).
#     A movie tower that turns movie titles into movie embeddings.
#     A loss that maximizes the predicted user-movie affinity for watches we observed, and minimizes it for watches that did not happen.
# TFRS and Keras provide a lot of the building blocks to make this happen. We can start with creating a model class. In the __init__ method, we set up some hyper-parameters as well as the primary components of the model.


class MovielensModel(tfrs.Model):
    """MovieLens prediction model."""

    def __init__(self):
        # The `__init__` method sets up the model architecture.
        super().__init__()

        # How large the representation vectors are for inputs: larger vectors make
        # for a more expressive model but may cause over-fitting.
        embedding_dim = 32
        num_unique_users = 500
        num_unique_movies = 1000
        eval_batch_size = 128

# The first major component is the user model: a set of layers that describe how raw user features should be transformed into numerical user representations. Here, we use the Keras preprocessing layers to turn user ids into integer indices, then map those into learned embedding vectors:

     # Set up user and movie representations.
        self.user_model = tf.keras.Sequential([
            # We first turn the raw user ids into contiguous integers by looking them
            # up in a vocabulary.
            # Maps strings from a vocabulary to integer indices.
            tf.keras.layers.experimental.preprocessing.StringLookup(
                max_tokens=num_unique_users),
            # We then map the result into embedding vectors. [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
            tf.keras.layers.Embedding(num_unique_users, embedding_dim)
        ])
        self.movie_model = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                max_tokens=num_unique_movies),
            tf.keras.layers.Embedding(num_unique_movies, embedding_dim)
        ])

# Once we have both user and movie models we need to define our objective and its evaluation metrics. In TFRS, we can do this via the Retrieval task (using the in-batch softmax loss):

        # The `Task` objects has two purposes: (1) it computes the loss and (2)
        # keeps track of metrics.
        self.task = tfrs.tasks.Retrieval(
            # In this case, our metrics are top-k metrics: given a user and a known
            # watched movie, how highly would the model rank the true movie out of
            # all possible movies?
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(eval_batch_size).map(self.movie_model)
            )
        )


# We use the compute_loss method to describe how the model should be trained.


    def compute_loss(self, features, training=False):
        # The `compute_loss` method determines how loss is computed.

        # Compute user and item embeddings.
        user_embeddings = self.user_model(features["user_id"])
        movie_embeddings = self.movie_model(features["movie_title"])

        # Pass them into the task to get the resulting loss. The lower the loss is, the
        # better the model is at telling apart true watches from watches that did
        # not happen in the training data.
        return self.task(user_embeddings, movie_embeddings)

# We can fit this model using standard Keras fit calls:


model = MovielensModel()
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

model.fit(ratings.batch(4096), verbose=False)

# To sanity-check the model’s recommendations we can use the TFRS BruteForce layer.
# The BruteForce layer is indexed with precomputed representations of candidates,
# and allows us to retrieve top movies in response to a query by computing the
# query-candidate score for all possible candidates:

index = tfrs.layers.ann.BruteForce(model.user_model)
index.index(movies.batch(100).map(model.movie_model), movies)

# Get recommendations.
_, titles = index(tf.constant(["42"]))

print('-------------------')

print(f"Recommendations for user 42: {titles[0, :3]}")

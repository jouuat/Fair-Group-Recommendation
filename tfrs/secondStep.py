
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tensorflow_recommenders as tfrs
import matplotlib.pyplot as plt
import tempfile
import os

plt.style.use('seaborn-whitegrid')

# En este tutorial usaremos los modelos del tutorial de características para generar incrustaciones.
# Por lo tanto, solo usaremos la identificación de usuario, la marca de tiempo y las características del título de la película.

ratings = tfds.load("movie_lens/100k-ratings", split="train")
movies = tfds.load("movie_lens/100k-movies", split="train")

ratings = ratings.map(lambda x: {
    "movie_title": x["movie_title"],
    "user_id": x["user_id"],
    "timestamp": x["timestamp"],
})
movies = movies.map(lambda x: x["movie_title"])

# También hacemos algunas tareas de limpieza para preparar vocabularios de características.

# Join a sequence of arrays along an existing axis [timestamp1,timestamp2,...]
timestamps = np.concatenate(list(ratings.map(lambda x: x["timestamp"]).batch(100)))

max_timestamp = timestamps.max()
min_timestamp = timestamps.min()

timestamp_buckets = np.linspace(
    min_timestamp, max_timestamp, num=1000,  # 1000 timestamp numeros entre el max i el min
)

# [movie1, movie2...](sense repeticions)
unique_movie_titles = np.unique(np.concatenate(list(movies.batch(1000))))
unique_user_ids = np.unique(np.concatenate(list(ratings.batch(1000).map(
    lambda x: x["user_id"]))))

# Definición de modelo
# Modelo de consulta
# Sera la primera capa del nostre model encarregat de convertir les entrades a incrustacions de caracteristiques


class UserModel(tf.keras.Model):
    # usermodel sera una subclasse del tf.keras.model i eradara tots els metodes i parametres mes el que afegim
    # super() fara que si criedem la classe tf.keras.model tambe poguem fer servir els metodes de la subclasse usermodel()
    def __init__(self):
        super().__init__()

        self.user_embedding = tf.keras.Sequential([  # Sequential agrupa un conjunt de linear de capes en un tf.keras.Model.
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_user_ids, mask_token=None),  # Mapeja els strings del unique_user_ids a indexs integers
            # (input size, numero de numeros amb vectors al que es transformara) exempl: [[4], [20]] -> [[0.25, 0.1], [0.6, -0.2]]
            tf.keras.layers.Embedding(len(unique_user_ids) + 1, 32),

        ])
        self.timestamp_embedding = tf.keras.Sequential([
            # ex; amb un input com aquest [1,2.9,2.3] ens dira l index al rang que pertany, (tenint en compte rangs [1,2),[1,3)]) ens tornaria la llista [1,2,2]
            tf.keras.layers.experimental.preprocessing.Discretization(timestamp_buckets.tolist()),
            # ens interessa nomes l ordre per saber si va ser recent o no, no el valor exacte
            tf.keras.layers.Embedding(len(timestamp_buckets) + 1, 32),
        ])
        # This layer will coerce its inputs into a distribution centered around 0 with standard deviation 1. (input-mean)/sqrt(var) at runtime.
        self.normalized_timestamp = tf.keras.layers.experimental.preprocessing.Normalization()
        # s'ha de cridar la funcio adapt perque funcioni la Normalization
        self.normalized_timestamp.adapt(timestamps)

    def call(self, inputs):  # quan fem una subclasse del model em d'implemetar el passe directe a la classe CALL
        # Take the input dictionary, pass it through each input layer,
        # and concatenate the result.
        print("--user_model_done--")
        return tf.concat([
            self.user_embedding(inputs["user_id"]),
            self.timestamp_embedding(inputs["timestamp"]),
            self.normalized_timestamp(inputs["timestamp"]),
        ], axis=1)  # [embeduser1.0...1.32, embeduser2...],[embedtimestamp1, embedtimestamp2...],[embednormaltimestamp1, embednormaltimestamp2]
# losses and metrics s'especificaran més tard a model.compile(),


# La definición de modelos más profundos requerirá que apilemos capas de modo encima de esta primera entrada.
# Una pila de capas cada vez más estrecha, separada por una función de activación, es un patrón común:

#                            +----------------------+
#                            |      128 x 64        |
#                            +----------------------+
#                                       | relu
#                          +--------------------------+
#                          |        256 x 128         |
#                          +--------------------------+
#                                       | relu
#                        +------------------------------+
#                        |          ... x 256           |
#                        +------------------------------+

# Para facilitar la experimentación con diferentes profundidades, definamos un modelo cuya profundidad (y ancho) está definida por un conjunto de parámetros de constructor.
# La capa oculta final no utiliza ninguna función de activación:
# el uso de una función de activación limitaría el espacio de salida de las incrustaciones finales y podría afectar negativamente
# el rendimiento del modelo.


class QueryModel(tf.keras.Model):
    """Model for encoding user queries."""

    def __init__(self, layer_sizes):
        """Model for encoding user queries.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        # We first use the user model for generating embeddings.
        self.embedding_model = UserModel()

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
        # tf.print(feature_embedding)
        print("-----query_model_done------")
        return self.dense_layers(feature_embedding)

# Modelo candidato
# Podemos adoptar el mismo enfoque para el modelo de película. Una vez más, comenzamos con la MovieModel del featurization tutorial:


class MovieModel(tf.keras.Model):

    def __init__(self):
        super().__init__()

        max_tokens = 10_000

        self.title_embedding = tf.keras.Sequential([
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=unique_movie_titles, mask_token=None),
            tf.keras.layers.Embedding(len(unique_movie_titles) + 1, 32)
        ])

        self.title_vectorizer = tf.keras.layers.experimental.preprocessing.TextVectorization(
            max_tokens=max_tokens)

        self.title_text_embedding = tf.keras.Sequential([
            self.title_vectorizer,
            tf.keras.layers.Embedding(max_tokens, 32, mask_zero=True),
            tf.keras.layers.GlobalAveragePooling1D(),
        ])

        self.title_vectorizer.adapt(movies)

    def call(self, titles):
        print("--movie_model--")
        return tf.concat([
            self.title_embedding(titles),
            self.title_text_embedding(titles),
        ], axis=1)

# Y amplíelo con capas ocultas:


class CandidateModel(tf.keras.Model):
    """Model for encoding movies."""

    def __init__(self, layer_sizes):
        """Model for encoding movies.

        Args:
          layer_sizes:
            A list of integers where the i-th entry represents the number of units
            the i-th layer contains.
        """
        super().__init__()

        self.embedding_model = MovieModel()

        # Then construct the layers.
        self.dense_layers = tf.keras.Sequential()

        # Use the ReLU activation for all but the last layer.
        for layer_size in layer_sizes[:-1]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size, activation="relu"))

        # No activation for the last layer.
        for layer_size in layer_sizes[-1:]:
            self.dense_layers.add(tf.keras.layers.Dense(layer_size))

    def call(self, inputs):
        feature_embedding = self.embedding_model(inputs)  # input del model
        print("-----candidate_model_done------")
        return self.dense_layers(feature_embedding)  # output del model

# Modelo combinado
# Con QueryModel y CandidateModel definidos, podemos armar un modelo combinado e implementar
# nuestra lógica de pérdidas y métricas. Para simplificar las cosas, exigiremos que la
# estructura del modelo sea la misma en todos los modelos de consulta y candidato.


class MovielensModel(tfrs.models.Model):

    def __init__(self, layer_sizes):
        super().__init__()
        self.query_model = QueryModel(layer_sizes)
        self.candidate_model = CandidateModel(layer_sizes)
        self.task = tfrs.tasks.Retrieval(
            metrics=tfrs.metrics.FactorizedTopK(
                candidates=movies.batch(128).map(self.candidate_model),
            ),
        )

    def compute_loss(self, features, training=False):
        # We only pass the user id and timestamp features into the query model. This
        # is to ensure that the training inputs would have the same keys as the
        # query inputs. Otherwise the discrepancy in input structure would cause an
        # error when loading the query model after saving it.
        query_embeddings = self.query_model({
            "user_id": features["user_id"],
            "timestamp": features["timestamp"],
        })
        # tf.print(query_embeddings) -> [[0.0659341887 0.106706105 0.0535111055 ... -0.0137582216 0.100897178 0.0968484357][0.0383222252 0.0228399653 -0.0141882207 ... -0.00664681336 -0.03621253 -0.0123816663][-0.157929257 -0.247049928 -0.386220306 ... 0.0821118057 -0.335981101 -0.231699213]...
        movie_embeddings = self.candidate_model(features["movie_title"])

        return self.task(
            query_embeddings, movie_embeddings, evaluate_metrics=not training)

# Entrenando el modelo
# Prepara los datos
# Primero dividimos los datos en un conjunto de entrenamiento y un conjunto de prueba.


tf.random.set_seed(42)
# ratings:["user_id:","movieid:", "rating:"] sense encoding
shuffled = ratings.shuffle(100_000, seed=42, reshuffle_each_iteration=False)

train = shuffled.take(80_000)
test = shuffled.skip(80_000).take(20_000)

cached_train = train.shuffle(100_000).batch(2048)
cached_test = test.batch(4096).cache()

# Modelo poco profundo
# ¡Estamos listos para probar nuestro primer modelo superficial!

num_epochs = 300
print("----- MovielensModel([32])------")
model = MovielensModel([32])
print("----- model.compile()------")
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))
print("----- model compiled------")
one_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=num_epochs,
    verbose=0)
print("----- model trained------")
print("---------- 1 layer model -------------")

accuracy = one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")

# Top-100 accuracy: 0.27.


# Esto nos da una precisión entre los 100 mejores de alrededor de 0,27. Podemos usar esto como un punto de referencia para evaluar modelos más profundos.

# Modelo más profundo
# ¿Qué tal un modelo más profundo con dos capas?

model = MovielensModel([64, 32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

two_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=num_epochs,
    verbose=0)

print("---------- 2 layer model -------------")

accuracy = two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")

# Top-100 accuracy: 0.29.


# La precisión aquí es 0.29, bastante mejor que el modelo poco profundo.

# Podemos trazar las curvas de precisión de validación para ilustrar esto:

'''num_validation_runs = len(one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"])
epochs = [(x + 1)* 5 for x in range(num_validation_runs)]

plt.plot(epochs, one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="1 layer")
plt.plot(epochs, two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="2 layers")
plt.title("Accuracy vs epoch")
plt.xlabel("epoch")
plt.ylabel("Top-100 accuracy");
plt.legend()

<matplotlib.legend.Legend at 0x7fb2cc120b70>'''


# Incluso al principio del entrenamiento, el modelo más grande tiene una ventaja clara y estable sobre el modelo superficial, lo que sugiere que agregar profundidad ayuda al modelo a capturar relaciones más matizadas en los datos.

# Sin embargo, los modelos aún más profundos no son necesariamente mejores. El siguiente modelo extiende la profundidad a tres capas:

model = MovielensModel([128, 64, 32])
model.compile(optimizer=tf.keras.optimizers.Adagrad(0.1))

three_layer_history = model.fit(
    cached_train,
    validation_data=cached_test,
    validation_freq=5,
    epochs=num_epochs,
    verbose=0)

print("---------- 3 layer model -------------")

accuracy = three_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"][-1]
print(f"Top-100 accuracy: {accuracy:.2f}.")


# De hecho, no vemos mejoras con respecto al modelo superficial:

print("---------- Comparison -------------")

plt.plot(
    epochs, one_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="1 layer")
plt.plot(
    epochs, two_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="2 layers")
plt.plot(
    epochs, three_layer_history.history["val_factorized_top_k/top_100_categorical_accuracy"], label="3 layers")
plt.title("Accuracy vs epoch")
plt.xlabel("epoch")
plt.ylabel("Top-100 accuracy")
plt.legend()

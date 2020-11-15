import tensorflow_datasets as tfds


class dataset:

    def __init__(self, config):
        self.dataset = config.dataset
        self.ratings = 0
        self.movies = 0
        self.movie_genres = 0
        self.bucketized_user_age = 0
        self.movie_title = 0
        self.user_id = 0
        self.timestamp = 0
        self.user_gender = 0
        self.user_occupation_text = 0
        self.user_rating = 0
        self.user_zip_code = 0

    def getData(self):
        if self.dataset.lower() == "tfds_movie_lens_100k":
            # {'bucketized_user_age': 45.0, 'movie_genres': array([7]), 'movie_id': b'357', 'movie_title': b"One Flew Over the Cuckoo's Nest (1975)", 'raw_user_age': 46.0, 'timestamp': 879024327, 'user_gender': True, 'user_id': b'138', 'user_occupation_label': 4, 'user_occupation_text': b'doctor', 'user_rating': 4.0, 'user_zip_code': b'53211'}
            self.ratings = tfds.load("movie_lens/100k-ratings", split="train")
            # {'movie_genres': array([4]), 'movie_id': b'1681', 'movie_title': b'You So Crazy (1994)'}
            self.movies = tfds.load("movie_lens/100k-movies", split="train")
            self.movie_genres = self.ratings.map(lambda x: x["movie_genres"])
            return self.ratings, self.movies
        else:
            print("no dataset selected")

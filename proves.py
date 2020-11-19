import pandas as pd

# relMatrix
data = {'movie_title': ['Film1', 'Film2', 'Film3'],
        'User1': [3, 4, 5],
        'User2': [5, 4, 3],
        }
relMatrix = pd.DataFrame(data, columns=['movie_title', 'User1', 'User2'])
relMatrix = relMatrix.set_index("movie_title")
print (relMatrix)

# ratings_pd
data = {'movie_title': ['Film1', 'Film2', 'Film3'],
        'user_id': ['User1', 'User2', 'User3'],
        'user_rating': [3, 4, 5],
        }
ratings_pd = pd.DataFrame(data, columns=['movie_title', 'user_id', 'user_rating'])
print (ratings_pd)
print(ratings_pd['user_id'] == 'User1')
print(ratings_pd['movie_title'] == 'Film1')
print(((ratings_pd['movie_title'] == 'Film1') & (ratings_pd['movie_title'] == 'Film1')).any())

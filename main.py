import pandas as pd
import numpy as np
import recommenders as rec

userInput = [
    # {'title': 'Mr. Nobody', 'rating': 5},
    {'title': 'The Godfather: Part II', 'rating': 5},
    {'title': 'GoodFellas', 'rating': 5},
    {'title': 'The Departed', 'rating': 5}
]
input_movies = pd.DataFrame(userInput)
user_titles = input_movies['title'].tolist()

# keyword_movies = rec.keyword_recommender(user_titles)
# # keyword_movies = movies[movies['title'].isin(recommended)]
#
# print(keyword_movies['title'].head())
#
# #movies = movies[movies['title'].isin(recommended)]
# genre_recs = rec.genre_recommender(input_movies, keyword_movies)
#
# print(genre_recs['title'].tolist())

# key_m = rec.keyword_recommender(['Mr. Nobody'], cosine_sim=rec.count_vectorizer())
# print(key_m['title'].head())
keyword_movies = rec.keyword_recommender(user_titles)
print(keyword_movies['title'].head())

# genre_recs = rec.genre_recommender(input_movies, key_m)
# print(genre_recs['title'].tolist())
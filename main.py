import pandas as pd
import numpy as np
import recommenders as rec

# dummy input
userInput = [
    {'title': 'The Godfather', 'rating': 5},
    {'title': 'The Godfather: Part II', 'rating': 5},
    {'title': 'GoodFellas', 'rating': 5},
    {'title': 'The Departed', 'rating': 5}
]
input_movies = pd.DataFrame(userInput)
user_titles = input_movies['title'].tolist()

# make a keyword recommendation using sum of 2 cosine similarities
# first cosine similarity: keywords extracted from movie overviews
# second cosine similarity: keywords from imdb, genres, director name
keyword_movies = rec.keyword_recommender(user_titles)

# sort the recommendations based on one hot encoding for genres and user rating
# genre_recs = rec.genre_recommender(input_movies, keyword_movies)
genre_recs = rec.genre_recommender(input_movies, keyword_movies)

# add column containing weighted ratings
weighted_movies = rec.set_weighted_rating(genre_recs)

# sort the movies by sum of weighted rating score + keywords score
sorted_movies = rec.summed_rating(weighted_movies)[['title', 'score', 'keyword_scores', 'summed']]
print(sorted_movies)

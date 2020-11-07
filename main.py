import pandas as pd
import numpy as np
import recommenders as rec

# dummy input
userInput = [
    {'title': 'The Godfather', 'rating': 5},
    {'title': 'The Godfather: Part II', 'rating': 5},
    {'title': 'GoodFellas', 'rating': 5},
    {'title': 'The Departed', 'rating': 5}
    # {'title': 'Her', 'rating': 5}
]
input_movies = pd.DataFrame(userInput)
user_titles = input_movies['title'].tolist()


def calculate(ids):
    user_titles = rec.get_titles_with_ids(ids)
    keyword_movies = rec.keyword_recommender(user_titles)
    genre_recs = rec.genre_recommender(input_movies, keyword_movies)
    weighted_movies = rec.set_weighted_rating(genre_recs)
    return weighted_movies


# make a keyword recommendation using sum of 2 cosine similarities
# first cosine similarity: keywords extracted from movie overviews
# second cosine similarity: keywords from imdb, genres, director name
# keyword_movies = rec.keyword_recommender(user_titles)

# sort the recommendations based on one hot encoding for genres and user rating
# genre_recs = rec.genre_recommender(input_movies, keyword_movies)
# genre_recs = rec.genre_recommender(input_movies, keyword_movies)

# add column containing weighted ratings
# weighted_movies = rec.set_weighted_rating(genre_recs)

# sort the movies by sum of weighted rating score + keywords score
# sorted_movies = rec.summed_rating(weighted_movies)[['title', 'score', 'keyword_scores', 'summed']]
# print(sorted_movies)

# print(rec.get_titles_with_ids([680, 61]))

final = pd.DataFrame()
correlated = rec.group_movies(user_titles)
while True:
    print(correlated.index)
    if len(correlated.index) == 0:
        break
    if len(correlated.index) <= 10:
        recommendation = calculate(correlated.index)
        # user_titles = rec.get_titles_with_ids(correlated.index)
        # keyword_movies = rec.keyword_recommender(user_titles)
        # genre_recs = rec.genre_recommender(input_movies, keyword_movies)
        # weighted_movies = rec.set_weighted_rating(genre_recs)
        final = final.append(recommendation)
        break
    grouped = correlated.sort_values(by=[correlated.index[0]], ascending=False)
    grouped = grouped[:10]
    gr_ids = grouped.index
    recommendation = calculate(gr_ids)
    final = final.append(recommendation)
    print(gr_ids)
    for i in gr_ids:
        correlated.drop(i, axis=0, inplace=True)
        correlated.drop(i, axis=1, inplace=True)

# user_titles = rec.get_titles_with_ids(correlated.index)
# print(user_titles)
# keyword_movies = rec.keyword_recommender(user_titles)
# genre_recs = rec.genre_recommender(input_movies, keyword_movies)
# weighted_movies = rec.set_weighted_rating(genre_recs)
# print(weighted_movies)
# final = final.append(weighted_movies)
# print(final)
# print(len(final.index))

clean = final.reset_index()
print(clean.head())
clean = clean.groupby('id').agg({'index': 'first',
                                 'title': 'first',
                                 'score': 'sum',
                                 'keyword_scores': 'sum'}).reset_index()
# print(clean)
# print(len(clean.index))
clean_sorted = rec.summed_rating(clean)[['title', 'score', 'keyword_scores', 'summed']]
print(clean_sorted)

import pandas as pd
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


# calculates recommendation for given ids
def calculate(ids):
    # get the titles for the current chunk of ids
    titles = rec.get_titles_from_ids(ids)
    # get 100 movies based on keyword similarity
    keyword_movies = rec.keyword_recommender(titles)
    # narrow down the 100 movies to 20 based on genres
    genre_recs = rec.genre_recommender(input_movies, keyword_movies)
    # set up a popularity rating column
    weighted_movies = rec.set_weighted_rating(genre_recs)
    return weighted_movies


# create empty dataframe to which we will append every recommendations per 10 movies
final = pd.DataFrame()
# the magic
correlated = rec.group_movies(user_titles)
while True:
    # if there's no more movies to go through, end the loop
    if len(correlated.index) == 0:
        break
    # calculate recommendations for the last chunk of the movies
    if len(correlated.index) <= 10:
        recommendation = calculate(correlated.index)
        # add recommendations to the dataframe
        final = final.append(recommendation)
        break
    #
    grouped = correlated.sort_values(by=[correlated.index[0]], ascending=False)
    # take 10 user movies
    grouped = grouped[:10]
    # calculate recommendations using the indexes for the movies
    gr_ids = grouped.index
    recommendation = calculate(gr_ids)
    # add recommendations to the dataframe
    final = final.append(recommendation)
    # drop the user movies that we already used
    for i in gr_ids:
        correlated.drop(i, axis=0, inplace=True)
        correlated.drop(i, axis=1, inplace=True)

# save the indexes in a column
clean = final.reset_index()
# the final recommendations dataframe can contain
# duplicate recommendations because it recommends for groups of 10 movies
# if we group them by id and sum the scores (eg: Interstellar shows up twice, we sum the keyword scores it has)
# the movies that showed up multiple times in the recommendations will get a higher value
clean = clean.groupby('id').agg({'index': 'first',
                                 'title': 'first',
                                 'score': 'first',
                                 'keyword_scores': 'sum'}).reset_index()

# final step is sum of the popularity rating for the movie and the final keyword score
# we get the final summed values for popularity and keyword scores sorted by highest value
clean_sorted = rec.summed_rating(clean)[['title', 'score', 'keyword_scores', 'summed']]
# print the final recommendations
print(clean_sorted)

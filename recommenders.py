import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler


# creates cosine similarity matrix based on keywords from the overview dataset
def vectorizer():
    # read the overviews for the movies
    # we use this for extracting keywords for each movie
    overview = pd.read_csv('datasets/overview_10k.csv')
    tfidf = TfidfVectorizer(stop_words='english')
    overview['overview'] = overview['overview'].fillna('')
    # Construct the required TF-IDF matrix by fitting and transforming the data
    tfidf_matrix = tfidf.fit_transform(overview['overview'])
    # Compute the cosine similarity matrix
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
    cosine_sim = cosine_sim + count_vectorizer()
    return cosine_sim


# creates cosine similarity matrices based on keywords from the keyword dataset or genres&director(soup) dataset
def count_vectorizer():
    cosine_sim = np.load('matrices/final_matrix.npy')
    return cosine_sim


# magic
def group_movies(user_titles):
    movies_db = pd.read_csv('datasets/final_movies.csv')
    cosine_sim = np.load('matrices/final_matrix.npy')
    ids = movies_db.loc[movies_db['title'].isin(user_titles)].index.tolist()
    similarities = []
    for id in ids:
        similarities.append(list(cosine_sim[id]))
    arr = np.array(similarities)
    arr2 = []
    for i in range(len(arr)):
        arr2.append(arr[i][ids])
    df = pd.DataFrame()
    i = 0
    for id in ids:
        df[id] = arr2[i]
        i = i + 1

    correlated = df.corr()
    return correlated


# Function that takes in movie titles, cosine similarity for similarity between movies and
# the movie database
# cosine similarity is a 10000x10000 matrix (10k because we have 10k movies in the dataset)
def keyword_recommender(titles, cosine_sim=vectorizer()):
    # read the movies database
    movies_db = pd.read_csv('datasets/final_movies.csv')
    # movies_db = movies_db[:4000]

    # cosine_sim = cosine_sim + count_vectorizer()
    # we create an array with zeroes and 10k elements
    # each row represents each movie
    final_scores = np.zeros(10000)
    # final_scores = np.zeros(4000)
    # we get the ids for the titles of the movies
    ids = movies_db.loc[movies_db['title'].isin(titles)].index.tolist()
    # we add rows from the cosine similarity matrix for each
    # movie ID to the empty array we created
    for id in ids:
        final_scores = np.add(final_scores, cosine_sim[id])

    # Get the summed similarity scores for all movies
    sim_scores = list(enumerate(final_scores))

    # Sort the movies based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 100 most similar movies
    sim_scores = sim_scores[0:100]

    # Get the indices of the 100 recommended movies
    movie_indices = [i[0] for i in sim_scores]
    # the list of recommended movies might contain
    # some of the movies that were already seen
    # movie_indices = list(set(movie_indices) - set(ids))

    movies_df = movies_db.iloc[movie_indices].copy()
    scores_arr = [i[1] for i in sim_scores]
    # print(scores_arr)
    movies_df['keyword_scores'] = scores_arr
    # this results contains the movies that the user has already seen
    # (they will have the highest ratings) & we have to remove them
    movies_df = movies_df[~movies_df.index.isin(ids)]

    return movies_df


# mean is calculated by mean = movies['vote_average'].mean()
def weighted_rating(x, mean=6.3, n_votes=160):
    v = x['vote_count']
    R = x['vote_average']
    # Calculation based on the IMDB formula
    return (v / (v + n_votes) * R) + (n_votes / (n_votes + v) * mean)


def set_weighted_rating(movies):
    # movies_db = pd.read_csv('datasets/final_movies.csv')
    movies['score'] = movies.apply(weighted_rating, axis=1)
    # movies = movies.sort_values('score', ascending=False)
    return movies


# sorts movies by sum of weighted ratings score & keyword score
def summed_rating(movies):
    scaler = MinMaxScaler(feature_range=(0, 1))
    movies['score'] = scaler.fit_transform(movies['score'].values.reshape(-1, 1))
    movies['summed'] = movies['score'] + movies['keyword_scores']
    movies = movies.sort_values('summed', ascending=False)
    return movies


def get_titles_with_ids(ids):
    movies_db = pd.read_csv('datasets/final_movies.csv')
    movies = movies_db.loc[ids, :]
    user_titles = movies['title'].tolist()
    return user_titles


def get_onehot(movies):
    onehot_db = pd.read_csv('datasets/genre_onehot_10k.csv')
    ids = movies['id'].array
    onehot_movies = onehot_db[onehot_db['id'].isin(ids)]
    return onehot_movies


def genre_recommender(user_movies, movies):
    # read the movies database
    movies_db = pd.read_csv('datasets/final_movies.csv')
    # find the user's movies in the movie database
    input_movies = movies_db[movies_db['title'].isin(user_movies['title'].tolist())]
    # get onehot values for each movie's genres
    user_onehot = get_onehot(input_movies)
    # reset the indexes and drop the id, we only need the columns with the genres
    user_onehot.reset_index(drop=True, inplace=True)
    user_onehot = user_onehot.drop(['id'], axis=1)
    # transpose to get values for genres based on the user's ratings
    user_profile = user_onehot.transpose().dot(user_movies['rating'])

    # get onehot values for the movies in the database
    # set the ID column as indices and drop it
    movies_onehot = get_onehot(movies)
    movies_onehot.reset_index(drop=True, inplace=True)
    movies_onehot = movies_onehot.set_index(movies_onehot['id'])
    movies_onehot = movies_onehot.drop(['id'], axis=1)

    # Multiply the genres by the weights and then take the weighted average
    final_rec = ((movies_onehot * user_profile).sum(axis=1)) / (user_profile.sum())
    # Sort our recommendations in descending order
    final_rec = final_rec.sort_values(ascending=False).head(20)
    # Find the titles of the recommended movies in the movie database
    final_movies = movies.loc[movies['id'].isin(final_rec.keys())]
    # return top 10 recommended movies
    return final_movies.head(20)

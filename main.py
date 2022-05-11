import streamlit as st

import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

books = pd.read_csv('BX-Books.csv', sep=';', error_bad_lines=False, encoding="latin-1")
books.columns = ['ISBN', 'bookTitle', 'bookAuthor', 'yearOfPublication', 'publisher', 'imageUrlS', 'imageUrlM',
                 'imageUrlL']

users = pd.read_csv('BX-Users.csv', sep=';', error_bad_lines=False, encoding="latin-1")
users.columns = ['userID', 'Location', 'Age']
ratings = pd.read_csv('BX-Book-Ratings.csv', sep=';', error_bad_lines=False, encoding="latin-1")
ratings.columns = ['userID', 'ISBN', 'bookRating']

# users with less than 200 ratings, and books with less than 100 ratings are excluded.

# counts1 = ratings['userID'].value_counts()
# ratings = ratings[ratings['userID'].isin(counts1[counts1 >= 200].index)]
# counts = ratings['bookRating'].value_counts()
# ratings = ratings[ratings['bookRating'].isin(counts[counts >= 100].index)]

combine_book_rating = pd.merge(ratings, books, on='ISBN')
columns = ['yearOfPublication', 'publisher', 'bookAuthor', 'imageUrlS', 'imageUrlM', 'imageUrlL']
combine_book_rating = combine_book_rating.drop(columns, axis=1)
# print(combine_book_rating)

# We then group by book titles and create a new column for total rating count.
combine_book_rating = combine_book_rating.dropna(axis=0, subset=['bookTitle'])

book_ratingCount = (combine_book_rating.
    groupby(by=['bookTitle'])['bookRating'].
    count().
    reset_index().
    rename(columns={'bookRating': 'totalRatingCount'})
[['bookTitle', 'totalRatingCount']]
    )
# print(book_ratingCount)

# We combine the rating data with the total rating count data, this gives us exactly what we need to find out which books are popular and filter out lesser-known books.
rating_with_totalRatingCount = combine_book_rating.merge(book_ratingCount, left_on='bookTitle', right_on='bookTitle',
                                                         how='left')
# print(rating_with_totalRatingCount)

pd.set_option('display.float_format', lambda x: '%.3f' % x)
# print(book_ratingCount['totalRatingCount'].describe())

# print(book_ratingCount['totalRatingCount'].quantile(np.arange(.9, 1, .01)))

popularity_threshold = 50
rating_popular_book = rating_with_totalRatingCount.query('totalRatingCount >= @popularity_threshold')
# print(rating_popular_book)

# print(rating_popular_book.shape)

combined = rating_popular_book.merge(users, left_on='userID', right_on='userID', how='left')

us_canada_user_rating = combined[combined['Location'].str.contains("usa|canada")]
us_canada_user_rating = us_canada_user_rating.drop('Age', axis=1)
# print(us_canada_user_rating)

# Implementing KNN
us_canada_user_rating = us_canada_user_rating.drop_duplicates(['userID', 'bookTitle'])
us_canada_user_rating_pivot = us_canada_user_rating.pivot(index='bookTitle', columns='userID',
                                                          values='bookRating').fillna(0)
us_canada_user_rating_matrix = csr_matrix(us_canada_user_rating_pivot.values)

model_knn = NearestNeighbors(metric='cosine', algorithm='brute')
model_knn.fit(us_canada_user_rating_matrix)

query_index = np.random.choice(us_canada_user_rating_pivot.shape[0])


def recommend(query_index):
    recommended_book_names = []
    us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1)
    distances, indices = model_knn.kneighbors(us_canada_user_rating_pivot.iloc[query_index, :].values.reshape(1, -1),
                                              n_neighbors=6)
    for i in range(0, len(distances.flatten())):
        if i == 0:
            recommended_book_names.append(us_canada_user_rating_pivot.index[query_index])
        else:
            recommended_book_names.append(us_canada_user_rating_pivot.index[indices.flatten()[i]])

    return recommended_book_names


print(query_index)
list = recommend(query_index)

book_list = []
st.header('Book Recommender System')
for i in range(1, 2442):
    book_list.append(us_canada_user_rating_pivot.index[i])

selected_movie = st.selectbox(
    "Type or select a book from the dropdown",
    book_list
)

if st.button('Show Recommendation'):
    col1 = st.columns(1)
    st.text(list[0])
    col2 = st.columns(1)
    st.text(list[1])
    col3 = st.columns(1)
    st.text(list[2])
    col4 = st.columns(1)
    st.text(list[3])
    col5 = st.columns(1)
    st.text(list[4])

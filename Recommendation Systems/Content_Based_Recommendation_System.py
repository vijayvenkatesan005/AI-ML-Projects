# Vijay Venkatesan

import pandas as pd
import numpy as np
import time
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

main_placeholder = st.empty()

movies_metadata_df = pd.read_csv('movies_metadata.csv')
movies_metadata_df = movies_metadata_df.iloc[:5000]

filtered_movies_metadata_df = movies_metadata_df[['original_title', 'overview', 'tagline', 'title']]

for column in ['original_title', 'overview', 'tagline', 'title']:
    filtered_movies_metadata_df[column] = filtered_movies_metadata_df[column].fillna('')

filtered_movies_metadata_df['combined_text'] = (
    filtered_movies_metadata_df['original_title'] + ' ' +
    filtered_movies_metadata_df['title'] + ' ' +
    filtered_movies_metadata_df['overview'] + ' ' +
    filtered_movies_metadata_df['tagline']
)

tfv = TfidfVectorizer(
    min_df=4,
    max_features=None,
    strip_accents='unicode',
    analyzer='word',
    token_pattern=r'\w{1,}',
    ngram_range=(1, 4),
    stop_words='english'
)

document_term_matrix = tfv.fit_transform(
    filtered_movies_metadata_df['combined_text']
)

main_placeholder.text("Computing Similarity Matrix...✅✅✅")
cosine_similarity_matrix = cosine_similarity(document_term_matrix, document_term_matrix)
time.sleep(2)
main_placeholder.empty()

movie_title_to_index_map = dict()
for index, movie_title in enumerate(filtered_movies_metadata_df['title']):
    movie_title_to_index_map[movie_title] = index

def get_recommendations(movie_title, top_k=5, cosine_similarity_matrix=cosine_similarity_matrix):
    
    similarity_scores_list = list(enumerate(cosine_similarity_matrix[movie_title_to_index_map[movie_title]]))

    similarity_scores_list = [(index, similarity_score) for index, similarity_score in similarity_scores_list if index != movie_title_to_index_map[movie_title]]
    
    sorted_similarity_scores_list = sorted(similarity_scores_list, key=lambda x: x[1], reverse=True)

    top_indices = [index for index, _ in sorted_similarity_scores_list[:top_k]]

    return filtered_movies_metadata_df['title'].iloc[top_indices] 

st.title("Movie Recommendation System 📽️🍿")
movie_title = st.text_input("Movie Title: ")
no_recommendations = st.text_input("Number of Recommendations: ")

if movie_title and no_recommendations:
    no_recommendations = int(no_recommendations)
    recommendations = get_recommendations(
        movie_title, 
        no_recommendations,
        cosine_similarity_matrix
    )

    st.header("Movie Recommendations")

    for recommendation in recommendations:
        st.write(recommendation)
    
    



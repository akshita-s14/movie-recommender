import streamlit as st
from model import recommend
import pandas as pd

movies = pd.read_csv('data/movies.csv')

st.set_page_config(page_title="Movie Recommender", layout="centered")

st.title("🎬 AI Movie Recommendation System")

st.write("Select a movie and get similar recommendations!")

movie_list = movies['title'].values
selected_movie = st.selectbox("Choose a movie", movie_list)

if st.button("Recommend"):
    recommendations = recommend(selected_movie)

    st.subheader("Recommended Movies:")
    for movie in recommendations:
        st.success(movie)
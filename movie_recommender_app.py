import streamlit as st
import pandas as pd
import numpy as np
import difflib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import joblib


# Custom CSS for advanced styling
def add_custom_css():
    st.markdown(
        """
        <style>
        /* General Background and Layout */
        body {
            background-color: #f4f4f9;
        }
        .main {
            background: #ffffff;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2);
        }
        h1 {
            color: #4A90E2;
            font-size: 2.8em;
            font-weight: bold;
        }
        h2, h3, h4 {
            color: #333333;
        }
        .stButton > button {
            background: linear-gradient(to right, #ff7e5f, #feb47b);
            color: white;
            border: none;
            border-radius: 10px;
            font-size: 1.2em;
            font-weight: bold;
            padding: 10px 25px;
            box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
            transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
        }
        .stButton > button:hover {
            background: linear-gradient(to right, #feb47b, #ff7e5f);
            transform: scale(1.05);
            box-shadow: 0px 6px 12px rgba(0, 0, 0, 0.3);
        }
        .stMarkdown {
            font-size: 1.2em;
        }
        .footer {
            text-align: center;
            margin-top: 50px;
            font-size: 0.9em;
            color: #555555;
        }
        </style>
        """,
        unsafe_allow_html=True
    )


# Function to load data and precomputed similarity matrix
@st.cache_data
def load_data():
    movies_data = pd.read_csv('processed_movies.csv')
    similarity = np.load('similarity_matrix.npy')
    return movies_data, similarity


# Function to get movie recommendations
def get_recommendations(movie_name, movies_data, similarity):
    # Get all movie titles
    list_of_all_titles = movies_data['title'].tolist()

    # Find closest match to user input
    find_close_match = difflib.get_close_matches(movie_name, list_of_all_titles)
    if not find_close_match:
        return None, []

    close_match = find_close_match[0]

    # Get index of the closest match
    index_of_the_movie = movies_data[movies_data.title == close_match]['index'].values[0]

    # Get similarity scores
    similarity_score = list(enumerate(similarity[index_of_the_movie]))

    # Sort movies based on similarity scores
    sorted_similar_movies = sorted(similarity_score, key=lambda x: x[1], reverse=True)

    # Get top 10 movie recommendations
    recommended_movies = []
    for movie in sorted_similar_movies[1:11]:  # Skip the first movie as it is the input movie itself
        index = movie[0]
        title_from_index = movies_data[movies_data.index == index]['title'].values[0]
        recommended_movies.append(title_from_index)

    return close_match, recommended_movies


# Main Streamlit app
def main():
    add_custom_css()  # Add custom styles

    # App Title and Introduction
    st.title("🎬 Movie Recommendation System")
    st.markdown(
        """
        Welcome to the **Movie Recommendation System**!  
        Discover movies similar to your favorites with the power of **Machine Learning**.  
        Just enter the name of a movie, and we'll suggest amazing recommendations for you!
        """,
        unsafe_allow_html=True
    )

    # Load data and precomputed similarity matrix
    movies_data, similarity = load_data()

    # User Input Section
    movie_name = st.text_input("🎥 Enter the name of a movie you like:")

    if st.button("🎉 Get Recommendations"):
        if movie_name.strip():
            close_match, recommendations = get_recommendations(movie_name, movies_data, similarity)

            if close_match:
                st.success(f"Movies similar to **{close_match}**:")
                for i, movie in enumerate(recommendations, 1):
                    st.write(f"**{i}.** {movie}")
            else:
                st.warning("Sorry, we couldn't find any matches. Try another movie name.")
        else:
            st.error("Please enter a valid movie name!")

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div class="footer">
        © 2025 | Made with ❤️ by Mohammed Jibril  
        Powered by **Streamlit**  
        </div>
        """,
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()

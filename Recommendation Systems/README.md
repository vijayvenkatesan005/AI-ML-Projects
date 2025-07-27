Movie Recommendation System 📽️🍿
This project is a content-based movie recommendation system built with Python and Streamlit. It leverages NLP techniques to suggest movies similar to a user-given movie title, using metadata from a film dataset.

Features
Content-Based Filtering: Uses TF-IDF vectorization on movie metadata (title, original_title, overview, tagline) to compute similarity between movies.

Cosine Similarity: Recommends movies based on text similarity with the selected film.

Interactive UI: Built with Streamlit, allowing users to quickly get recommendations for any movie in the dataset.

Handles Missing Data: Automatically deals with missing metadata to ensure smooth recommendations.

Efficient for Demos: Loads a manageable subset for quick prototyping and presentations.

Getting Started
Prerequisites
Python 3.7+

streamlit

pandas

scikit-learn

numpy

Install requirements:

bash
pip install streamlit pandas scikit-learn numpy
Dataset
Place a file named movies_metadata.csv in the project root. This should contain at least the columns: original_title, overview, tagline, title.

Running the App
bash
streamlit run your_script.py
Replace your_script.py with the filename containing your code.

How it Works
Data Preparation
Loads the first 5,000 entries from movies_metadata.csv and combines key metadata into one text field per movie. Missing values are filled to avoid errors.

Feature Extraction
Transforms the combined text using TF-IDF with n-grams and English stopword removal.

Similarity Calculation
Computes a movie-to-movie cosine similarity matrix for efficient lookup.

User Interaction

The user enters or selects a movie title and the desired number of recommendations.

The app displays a list of top similar movies.

Example Usage
python
# Command line
streamlit run app.py

# In the browser: 
# - Enter/select a movie title (e.g. "Toy Story")
# - Choose number of recommendations (e.g. 5)
# - Get relevant recommendations instantly
Project Structure
text
.
├── app.py                  # Main Streamlit app code
├── movies_metadata.csv     # Movie metadata dataset
└── README.md               # This file

License
This project is provided under the MIT License.

Feel free to adjust or extend this template according to your dataset, personal branding, or extra features!

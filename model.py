import pandas as pd
import ast
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load datasets
movies = pd.read_csv('data/movies.csv')
credits = pd.read_csv('data/credits.csv')

# Merge datasets
movies = movies.merge(credits, on='title')

# Select required columns
movies = movies[['title', 'overview', 'genres', 'keywords']].dropna()

# Convert JSON-like columns to text
def convert(text):
    L = []
    for i in ast.literal_eval(text):
        L.append(i['name'])
    return " ".join(L)

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)

# Combine all features
movies['tags'] = movies['overview'] + " " + movies['genres'] + " " + movies['keywords']

# Convert to lowercase
movies['tags'] = movies['tags'].apply(lambda x: x.lower())

# Vectorization
tfidf = TfidfVectorizer(stop_words='english')
vectors = tfidf.fit_transform(movies['tags'])

# Similarity
similarity = cosine_similarity(vectors)

def recommend(movie):
    movie = movie.lower()
    
    if movie not in movies['title'].str.lower().values:
        return ["Movie not found"]

    index = movies[movies['title'].str.lower() == movie].index[0]
    distances = similarity[index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:6]

    return [movies.iloc[i[0]].title for i in movies_list]
from flask import Flask, request, jsonify
import pickle
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask_cors import CORS

app = Flask(__name__)
CORS(app) 
# Load the dataset and models
with open('movies.pkl', 'rb') as f:
    dataset = pd.DataFrame.from_dict(pickle.load(f))

# Prepare data for recommendation
# Prepare data for recommendation
# Create a copy of the DataFrame
data = dataset[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']].copy()

# Combine all text fields into 'tags'
data['tags'] = data['title'] + " " + data['overview'] + " " + data['genres'].apply(lambda x: ' '.join(x)) + " " + data['keywords'].apply(lambda x: ' '.join(x)) + " " + data['cast'].apply(lambda x: ' '.join(x)) + " " + data['crew'].apply(lambda x: ' '.join(x))

# Convert tags to lowercase
data.loc[:, 'tags'] = data['tags'].str.lower()

# Handle NaN values
data.loc[:, 'tags'] = data['tags'].fillna('')


cv = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = cv.fit_transform(data['tags']).toarray()
similarity = cosine_similarity(vectors)


@app.route('/recommend', methods=['GET'])
def recommend():
    movie_title = request.args.get('title')
    if not movie_title:
        return jsonify({'error': 'Title parameter is required'}), 400
    
    try:
        movie_index = data[data['title'] == movie_title].index[0]
    except IndexError:
        return jsonify({'error': 'Movie not found'}), 404
    
    distances = similarity[movie_index]
    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x: x[1])[1:7]
    
    recommendations = [{'title': data.iloc[i[0]]['title'], 'similarity': i[1]} for i in movies_list]
    
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)

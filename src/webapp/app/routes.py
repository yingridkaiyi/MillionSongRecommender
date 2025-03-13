from flask import Blueprint, request, jsonify, render_template
import sys
import os
from pathlib import Path

# i had trouble with accessing my project paths, so I had to manually add the path
project_root = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(project_root))

from src.models.song_recommender import SongRecommender

main = Blueprint('main', __name__)
recommender = SongRecommender()  # Initialize the recommender

@main.route('/')
def index():
    # Pass available genres to the template
    genres = recommender.get_available_genres()
    return render_template('index.html', genres=genres)

@main.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_input = data.get('input')
    top_n = data.get('top_n', 10)
    genre = data.get('genre')  # Get genre from request

    # Call the recommender with genre
    recommendations = recommender.recommend_songs(user_input, genre=genre, top_n=top_n)
    return jsonify(recommendations)
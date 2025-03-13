from src.models.scenario_processor import ScenarioProcessor
from src.models.feature_matcher import FeatureMatcher
from src.database.db_manager import DatabaseManager


#test scnario processor
user_input = "I'm working out with friends in the late evening."
scenario_features = ScenarioProcessor().process_user_input(user_input)
print("Processed Scenario Features:")
print(scenario_features)

#test feature matcher
feature_ranges = FeatureMatcher().get_feature_ranges(scenario_features)
print("Feature Ranges:")
print(feature_ranges)

#song data
song_data = DatabaseManager().get_song_features(limit=10000)
print("Song_data:")
print(song_data.head())
print(song_data.columns)


def filter_songs(song_data, feature_ranges):
    # Create a copy of the song data
    filtered_songs = song_data.copy()
    
    # Initialize a score column
    filtered_songs['score'] = 0
    
    # Calculate the score for each song
    for feature, (min_val, max_val) in feature_ranges.items():
        filtered_songs['score'] += (
            (filtered_songs[feature] >= min_val) & 
            (filtered_songs[feature] <= max_val)
        ).astype(int)
    
    return filtered_songs


def recommend_songs(song_data, feature_ranges, top_n=5):
    # Filter and score songs
    scored_songs = filter_songs(song_data, feature_ranges)
    
    # Sort by score (descending) and select top N songs
    recommended_songs = scored_songs.sort_values(by='score', ascending=False).head(top_n)
    
    return recommended_songs

recommended_songs = recommend_songs(song_data, feature_ranges, top_n=5)
print("Recommended Songs:")
print(recommended_songs[['track_name', 'artist_name', 'score'] + list(feature_ranges.keys())])
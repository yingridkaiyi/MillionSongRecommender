import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from src.models.scenario_processor import ScenarioProcessor
from src.models.feature_matcher import FeatureMatcher
from src.database.db_manager import DatabaseManager

class SongRecommender:
    # Define genre characteristics based on audio features
    GENRE_PROFILES = {
        'electronic': {
            'danceability': (0.6, 1.0),
            'energy': (0.7, 1.0),
            'instrumentalness': (0.4, 1.0),
            'acousticness': (0.0, 0.3)
        },
        'acoustic': {
            'acousticness': (0.7, 1.0),
            'energy': (0.3, 0.7),
            'instrumentalness': (0.0, 0.4)
        },
        'hip_hop': {
            'speechiness': (0.2, 1.0),
            'danceability': (0.6, 1.0),
            'acousticness': (0.0, 0.4)
        },
        'classical': {
            'instrumentalness': (0.7, 1.0),
            'acousticness': (0.6, 1.0),
            'speechiness': (0.0, 0.1)
        },
        'rock': {
            'energy': (0.7, 1.0),
            'instrumentalness': (0.0, 0.3),
            'valence': (0.4, 0.8)
        }
    }

    def __init__(self):
        self.scenario_processor = ScenarioProcessor()
        self.feature_matcher = FeatureMatcher()
        self.db_manager = DatabaseManager()
        # Cache the song data
        self._song_data = None

    def _get_song_data(self):
        """Get or cache song data"""
        if self._song_data is None:
            self._song_data = self.db_manager.get_song_features()
        return self._song_data

    def recommend_songs(self, user_input: str, genre: str = None, top_n: int = 10) -> List[Dict]:
        scenario_features = self.scenario_processor.process_user_input(user_input)
        feature_ranges = self.feature_matcher.get_feature_ranges(scenario_features)
        
        # Get songs from cache
        song_data = self._get_song_data()
        
        # Apply genre filtering if specified
        if genre and genre in self.GENRE_PROFILES:
            song_data = self._apply_genre_filter(song_data, genre)
        
        # Score and sort songs more efficiently
        scored_songs = self._filter_and_score_songs(song_data, feature_ranges)
        recommend_songs = scored_songs.nlargest(top_n, 'score')

        # Convert to list of dictionaries
        return [{
            "track_name": row["track_name"],
            "artist_name": row["artist_name"],
            "danceability": row["danceability"],
            "energy": row["energy"],
            "valence": row["valence"],
            "tempo": row["tempo"],
            "acousticness": row["acousticness"],
            "instrumentalness": row["instrumentalness"]
        } for _, row in recommend_songs.iterrows()]

    def _apply_genre_filter(self, song_data: pd.DataFrame, genre: str) -> pd.DataFrame:
        genre_features = self.GENRE_PROFILES[genre]
        mask = pd.Series(True, index=song_data.index)
        
        for feature, (min_val, max_val) in genre_features.items():
            mask &= (song_data[feature] >= min_val) & (song_data[feature] <= max_val)
        
        return song_data[mask]

    def _filter_and_score_songs(self, song_data: pd.DataFrame, feature_ranges: Dict[str, Tuple[float, float]]) -> pd.DataFrame:
        # Calculate scores using vectorized operations
        scores = pd.Series(0, index=song_data.index)
        for feature, (min_val, max_val) in feature_ranges.items():
            if feature in song_data.columns:
                scores += ((song_data[feature] >= min_val) & 
                         (song_data[feature] <= max_val)).astype(int)
        
        # Add scores to the dataframe
        result = song_data.copy()
        result['score'] = scores
        return result

    def get_available_genres(self) -> List[str]:
        return list(self.GENRE_PROFILES.keys())
    
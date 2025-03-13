import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from src.models.scenario_processor import ScenarioProcessor

class FeatureMatcher:
    def __init__(self):
        self.feature_mappings()
        self.feature_scaler = MinMaxScaler
        self.feature_names = ['danceability', 'energy', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'valence', 'tempo']
        self.init_scenario_mappings()

    def feature_mappings(self):
        self.mood_features = {
                'happy': {
                'valence': (0.7, 0.9),
                'energy': (0.6, 0.8),
                'danceability': (0.6, 0.8)
            },
            'focused': {
                'instrumentalness': (0.6, 0.9),
                'energy': (0.4, 0.6),
                'speechiness': (0.0, 0.3)
            },
            'relaxed': {
                'energy': (0.2, 0.4),
                'acousticness': (0.6, 0.9),
                'instrumentalness': (0.4, 0.7)
            },
            'energetic': {
                'energy': (0.8, 1.0),
                'tempo': (120, 140),
                'valence': (0.6, 0.8)
            },
            'sad': {
                'valence': (0.1, 0.3),
                'energy': (0.2, 0.4),
                'tempo': (60, 90)
            }
        }

        self.activity_features = {
            'working': {
                'instrumentalness': (0.6, 0.9),
                'speechiness': (0.0, 0.3),
                'energy': (0.4, 0.6)
            },
            'exercising': {
                'energy': (0.8, 1.0),
                'tempo': (120, 140),
                'danceability': (0.6, 0.8)
            },
            'relaxing': {
                'energy': (0.2, 0.4),
                'acousticness': (0.6, 0.9),
                'instrumentalness': (0.4, 0.7)
            },
            'commuting': {
                'energy': (0.5, 0.7),
                'danceability': (0.5, 0.7),
                'valence': (0.5, 0.7)
            },
            'socializing': {
                'danceability': (0.7, 0.9),
                'energy': (0.7, 0.9),
                'valence': (0.7, 0.9)
            }
        }

        self.time_features = {
            'morning': {
                'energy': (0.5, 0.7),
                'valence': (0.5, 0.7),
                'tempo': (90, 120)
            },
            'afternoon': {
                'energy': (0.6, 0.8),
                'valence': (0.5, 0.7),
                'tempo': (100, 130)
            },
            'evening': {
                'energy': (0.4, 0.6),
                'acousticness': (0.5, 0.8),
                'tempo': (80, 110)
            },
            'night': {
                'energy': (0.3, 0.5),
                'acousticness': (0.6, 0.9),
                'tempo': (70, 100)
            }
        }

        self.social_features = {
            'alone': {
                'acousticness': (0.6, 0.9),
                'instrumentalness': (0.5, 0.8),
                'energy': (0.3, 0.5)
            },
            'with_friends': {
                'danceability': (0.7, 0.9),
                'energy': (0.7, 0.9),
                'valence': (0.7, 0.9)
            },
            'with_family': {
                'acousticness': (0.5, 0.8),
                'valence': (0.6, 0.8),
                'energy': (0.4, 0.6)
            }
        }
        
    def init_scenario_mappings(self):

        self.scenario_mappings = []
        scenario_processor = ScenarioProcessor()

        for mood in self.mood_features.keys():
            for activity in self.activity_features.keys():
                for time in self.time_features.keys():
                    for social in self.social_features.keys():
                        # Create scenario description
                        scenario_text = f"{activity} in the {time}, feeling {mood}"
                        if social != "unspecified":
                            scenario_text += f", {social}"
                        
                        # Process scenario to get embedding
                        features = scenario_processor.process_user_input(scenario_text)
                        
                        # Combine feature ranges
                        combined_features = {}
                        if mood in self.mood_features:
                            combined_features.update(self.mood_features[mood])
                        if activity in self.activity_features:
                            self._merge_ranges(combined_features, self.activity_features[activity])
                        if time in self.time_features:
                            self._merge_ranges(combined_features, self.time_features[time])
                        if social in self.social_features:
                            self._merge_ranges(combined_features, self.social_features[social])

                        self.scenario_mappings.append({
                            'mood': mood,
                            'activity': activity,
                            'time': time, 
                            'social': social,
                            'features': combined_features,
                            'embedding': features['embedding'],
                            'text': scenario_text
                        })

    def _merge_ranges(self, ranges:Dict, new_ranges: Dict):
        for feature, (min_val, max_val) in new_ranges.items():
            if feature in ranges: 
                current_min, current_max = ranges[feature]
                ranges[feature] =  (min(current_min, min_val), max(current_max, max_val))
            else: 
                ranges[feature] = (min_val, max_val)


    def get_feature_ranges(self, scenario_features: Dict) -> Dict[str, Tuple[float, float]]:
        feature_ranges = self._get_base_ranges(scenario_features)
        similar_scenarios = self._similar_scenarios(scenario_features)

        if not similar_scenarios:
            similar_scenarios = [{'scenario': self.default_scenario, 'similarity': 1.0}]

        if similar_scenarios: 
            for feature in self.feature_names: 
                similar_mins = []
                similar_maxs = []

                for similar in similar_scenarios: 
                    if feature in similar['scenario']['features']:
                        min_val, max_val = similar['scenario']['features'][feature]
                        similar_mins.append(min_val)
                        similar_maxs.append(max_val)

                    if similar_mins and similar_maxs:
                        if feature in feature_ranges:
                            current_min, current_max = feature_ranges[feature]
                            feature_ranges[feature] = (
                                (current_min + np.mean(similar_mins)) / 2,
                                (current_max + np.mean(similar_maxs)) / 2
                            )
                        else: 
                            feature_ranges[feature] = (np.mean(similar_mins), np.mean(similar_maxs))
                    
        return feature_ranges


    def _similar_scenarios(self, scenario_features: Dict) -> List[Dict]:
        """Find similar scenarios using cosine similarity"""
        # Reshape current embedding to 2D array (1, n_features)
        current_embedding = scenario_features['embedding'].reshape(1, -1)
        
        # Create 2D array of scenario embeddings (n_scenarios, n_features)
        scenario_embeddings = np.array([s['embedding'] for s in self.scenario_mappings])
        
        # Calculate similarities
        similarities = cosine_similarity(current_embedding, scenario_embeddings).flatten()
        
        # Create list of scenarios with their similarity scores
        similar_scenarios = []
        for i, similarity in enumerate(similarities):
            similar_scenarios.append({
                'scenario': self.scenario_mappings[i],
                'similarity': similarity
            })
        
        # Sort by similarity and return top 3
        similar_scenarios.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_scenarios[:3]

    def _get_base_ranges(self, scenario_features: Dict) -> Dict[str, Tuple[float, float]]:
        feature_ranges = {}

        if scenario_features['mood'] != 'unspecified':
            mood_ranges = self.mood_features.get(scenario_features['mood'], {})
            feature_ranges.update(mood_ranges)
        
        if scenario_features['activity'] != 'unspecified':
            activity_ranges = self.activity_features.get(scenario_features['activity'], {})
            self._merge_ranges(feature_ranges, activity_ranges)

        if scenario_features['time'] != 'unspecified':
            time_ranges = self.time_features.get(scenario_features['time'], {})
            self._merge_ranges(feature_ranges, time_ranges)

        if scenario_features['social'] != 'unspecified':
            social_ranges = self.social_features.get(scenario_features['social'], {})
            self._merge_ranges(feature_ranges, social_ranges)
        
        return feature_ranges
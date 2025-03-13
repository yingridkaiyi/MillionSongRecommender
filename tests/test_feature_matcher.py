# tests/test_feature_matcher.py

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.scenario_processor import ScenarioProcessor
from src.models.feature_matcher import FeatureMatcher

def test_feature_matcher():
    # Initialize
    scenario_processor = ScenarioProcessor()
    feature_matcher = FeatureMatcher()
    
    # Test scenarios
    test_scenarios = [
        
        "Working from home in the morning, feeling focused",
        "Morning workout at the gym, feeling energetic",
        
        
        "Studying in the library during afternoon",
        "Afternoon jog in the park",
        
        
        "Relaxing evening at home after work",
        "Evening party with friends",
        
        
        "Late night coding session",
        "Night time meditation",
        
        
        "Feeling sad while commuting in the evening",
        "Happy morning coffee with friends",
        
        
        "Just want some music",  
        "Feeling really energetic and focused while working out at the gym in the morning"  
    ]
    
    print("\nTesting Feature Matcher:")
    print("="*80)
    
    for scenario in test_scenarios:
        print(f"\nScenario: {scenario}")
        
        
        features = scenario_processor.process_user_input(scenario)
        print("\nExtracted Features:")
        for key, value in features.items():
            if key not in ['embedding', 'raw_text']:
                print(f"- {key}: {value}")
        
        
        ranges = feature_matcher.get_feature_ranges(features)
        
        print("\nFeature Ranges:")
        for feature, (min_val, max_val) in ranges.items():
            print(f"- {feature}: {min_val:.2f} - {max_val:.2f}")
        
        
        similar = feature_matcher._find_similar_scenarios(features)
        if similar:
            print("\nSimilar Scenarios:")
            for s in similar:
                print(f"- Mood: {s['scenario']['mood']}, "
                      f"Activity: {s['scenario']['activity']}, "
                      f"Time: {s['scenario']['time']}, "
                      f"Similarity: {s['similarity']:.2f}")
        
        print("-"*80)

def test_specific_cases():
    
    scenario_processor = ScenarioProcessor()
    feature_matcher = FeatureMatcher()
    
    print("\nTesting Specific Cases:")
    print("="*80)
    
    
    scenario = "Feeling happy"
    print(f"\nTest Case 1 - Only Mood: {scenario}")
    features = scenario_processor.process_user_input(scenario)
    ranges = feature_matcher.get_feature_ranges(features)
    print("Feature Ranges:")
    for feature, (min_val, max_val) in ranges.items():
        print(f"- {feature}: {min_val:.2f} - {max_val:.2f}")
    
    
    scenario = "Working"
    print(f"\nTest Case 2 - Only Activity: {scenario}")
    features = scenario_processor.process_user_input(scenario)
    ranges = feature_matcher.get_feature_ranges(features)
    print("Feature Ranges:")
    for feature, (min_val, max_val) in ranges.items():
        print(f"- {feature}: {min_val:.2f} - {max_val:.2f}")
    
   
    scenario = "In the morning"
    print(f"\nTest Case 3 - Only Time: {scenario}")
    features = scenario_processor.process_user_input(scenario)
    ranges = feature_matcher.get_feature_ranges(features)
    print("Feature Ranges:")
    for feature, (min_val, max_val) in ranges.items():
        print(f"- {feature}: {min_val:.2f} - {max_val:.2f}")
    
    
    scenario = "Feeling energetic while working out at the gym in the morning"
    print(f"\nTest Case 4 - Complex Scenario: {scenario}")
    features = scenario_processor.process_user_input(scenario)
    ranges = feature_matcher.get_feature_ranges(features)
    print("Feature Ranges:")
    for feature, (min_val, max_val) in ranges.items():
        print(f"- {feature}: {min_val:.2f} - {max_val:.2f}")
    
    print("-"*80)

if __name__ == "__main__":
    print("Running Feature Matcher Tests...")
    test_feature_matcher()
    test_specific_cases()
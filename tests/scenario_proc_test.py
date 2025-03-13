from src.models.scenario_processor import ScenarioProcessor

def test_scenario_proceessor():
    processor = ScenarioProcessor()

    test_scenarios = [
        "I'm working out from home in the morning, feeling focused", 
        "Running with friends in the evening, feeling energetic", 
        "Ralaxing alone a night, I am feeling quite happy",
        "I am partying with friends at midnight"
    ]

    for scenario in test_scenarios:
        features = processor.process_user_input(scenario)
        print(f"Scenario: {scenario}")
        print(f"Features:")
        for key, value in features.items():
            if key != 'embedding':
                print(f"-{key}:{value}")

if __name__ == "__main__":
    test_scenario_proceessor()
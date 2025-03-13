from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

class ScenarioProcessor:
    #since I want to keep this project at a manageable scale, i just decided to manually created
        #some commonly seen categories and scenarios (time, mood, activity, social)

    TIME_CATEGORIES = {
        'morning': ['morning', 'dawn', 'breakfast', 'early'],
        'afternoon': ['afternoon', 'lunch', 'midday', 'noon'],
        'evening': ['evening', 'sunset', 'dinner', 'dusk'],
        'night': ['night', 'late', 'midnight', 'bedtime']
    }
    
    ACTIVITY_CATEGORIES = {
        'working': ['working', 'studying', 'coding', 'writing', 'reading'],
        'exercising': ['workout', 'exercise', 'running', 'gym', 'training'],
        'relaxing': ['relaxing', 'chilling', 'resting', 'meditating'],
        'commuting': ['driving', 'commuting', 'traveling', 'walking'],
        'socializing': ['party', 'gathering', 'meeting', 'hangout']
    }
    
    MOOD_CATEGORIES = {
        'happy': ['happy', 'joyful', 'excited', 'cheerful'],
        'focused': ['focused', 'concentrated', 'productive'],
        'relaxed': ['relaxed', 'calm', 'peaceful', 'mellow'],
        'energetic': ['energetic', 'pumped', 'motivated'],
        'sad': ['sad', 'melancholic', 'down', 'blue']
    }
    
    SOCIAL_CONTEXT = {
        'alone': ['alone', 'solo', 'by myself'],
        'with_friends': ['friends', 'group', 'party'],
        'with_family': ['family', 'relatives', 'home']
    }

    def __init__(self):
        """
        TODO:
        Initialize:
        - DistilRoBERTa model and tokenizer
        - Scenario categories/features
        """
        self.tokenizer = AutoTokenizer.from_pretrained('distilroberta-base')
        self.model = AutoModel.from_pretrained('distilroberta-base')
        self.model.eval() # we need to tell the model that we are predicting, not training
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

    def process_user_input(self, input_text:str) -> dict:
        input_text = input_text.lower() 

        features = {
            'time': self._extract_time(input_text),
            'activity': self._extract_activity(input_text),
            'mood': self._extract_mood(input_text),
            'social': self._extract_social_context(input_text),
            'raw_text': input_text,
            'embedding': self.generate_embedding(input_text)
        }
        return features

    def _extract_time(self, text: str) -> str:
        
        for time, keywords in self.TIME_CATEGORIES.items():
            if any(keyword in text for keyword in keywords):
                return time
        return 'unspecified'

    def _extract_activity(self, text: str) -> str:
        
        for activity, keywords in self.ACTIVITY_CATEGORIES.items():
            if any(keyword in text for keyword in keywords):
                return activity
        return 'unspecified'

    def _extract_mood(self, text: str) -> str:
        
        for mood, keywords in self.MOOD_CATEGORIES.items():
            if any(keyword in text for keyword in keywords):
                return mood
        return 'unspecified'

    def _extract_social_context(self, text: str) -> str:
        
        for context, keywords in self.SOCIAL_CONTEXT.items():
            if any(keyword in text for keyword in keywords):
                return context
        return 'unspecified'

    def generate_embedding(self, text:str) -> np.ndarray:
        """
        TODO:
        Convert processed scenario into embeddings using DistilRoBERTa
        """
        inputs = self.tokenizer(text, 
                                return_tensors = "pt",
                                padding=True,
                                truncation=True,
                                max_length=512
                                )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        
        with torch.no_grad(): 
            outputs = self.model(**inputs)
            
        
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        
        return embedding[0]  # Return the first (and only) embedding

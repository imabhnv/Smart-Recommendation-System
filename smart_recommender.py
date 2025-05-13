# import pandas as pd
# import numpy as np
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# import nltk
# from nltk.sentiment import SentimentIntensityAnalyzer
# import joblib
# from datetime import datetime
# import random

# # Download NLTK resources for sentiment analysis
# nltk.download('vader_lexicon', quiet=True)

# class SmartRecommender:
#     def __init__(self):
#         """Initialize the Smart Recommendation System."""
#         self.places_df = None
#         self.users_df = None
#         self.ratings_df = None
#         self.content_matrix = None
#         self.user_item_matrix = None
#         self.sia = SentimentIntensityAnalyzer()
#         self.mood_weights = {
#             'very_negative': {'active': 0.1, 'relaxing': 0.9, 'popular': 0.3, 'hidden_gem': 0.7},
#             'negative': {'active': 0.3, 'relaxing': 0.7, 'popular': 0.4, 'hidden_gem': 0.6},
#             'neutral': {'active': 0.5, 'relaxing': 0.5, 'popular': 0.5, 'hidden_gem': 0.5},
#             'positive': {'active': 0.7, 'relaxing': 0.3, 'popular': 0.6, 'hidden_gem': 0.4},
#             'very_positive': {'active': 0.9, 'relaxing': 0.1, 'popular': 0.7, 'hidden_gem': 0.3}
#         }
        
#     def load_sample_data(self):
#         """Load sample data for demonstration purposes."""
#         # Sample data for places (restaurants, activities, etc.)
#         places_data = {
#             'place_id': list(range(1, 21)),
#             'name': [
#                 'Green Garden Restaurant', 'Sunset Beach', 'Mountain Trail Hike',
#                 'City Museum', 'Sky Diving Experience', 'Cozy Cafe', 
#                 'Historic Library', 'Botanical Gardens', 'Central Park',
#                 'Harbor Cruise', 'Rock Climbing Center', 'Downtown Cinema',
#                 'Art Gallery', 'Yoga Studio', 'Live Music Bar',
#                 'Lakeside Picnic Area', 'Adventure Park', 'Spa Retreat',
#                 'Local Brewery Tour', 'Theatre Performance'
#             ],
#             'category': [
#                 'restaurant', 'outdoor', 'outdoor', 'culture', 'adventure',
#                 'restaurant', 'culture', 'outdoor', 'outdoor', 'adventure',
#                 'adventure', 'entertainment', 'culture', 'wellness', 'entertainment',
#                 'outdoor', 'adventure', 'wellness', 'entertainment', 'culture'
#             ],
#             'description': [
#                 'Organic vegetarian restaurant with garden seating',
#                 'Beautiful beach with stunning sunset views',
#                 'Challenging mountain trail with scenic overlooks',
#                 'Expansive museum showcasing city history and art',
#                 'Thrilling sky diving experience for adrenaline seekers',
#                 'Intimate cafe with specialty coffees and pastries',
#                 'Historic library with rare book collection',
#                 'Extensive gardens featuring exotic plants and flowers',
#                 'Large urban park with walking paths and ponds',
#                 'Relaxing cruise around the harbor with city views',
#                 'Indoor climbing center with routes for all levels',
#                 'Modern cinema showing latest blockbusters',
#                 'Contemporary art gallery with rotating exhibits',
#                 'Peaceful yoga studio offering various classes',
#                 'Energetic bar featuring live local bands',
#                 'Serene picnic area by the lake with BBQ facilities',
#                 'Family adventure park with ziplines and obstacle courses',
#                 'Luxury spa offering massage and wellness treatments',
#                 'Guided tour of local craft breweries with tastings',
#                 'Theater showing acclaimed performances and plays'
#             ],
#             'price_level': [3, 1, 2, 2, 4, 2, 1, 2, 1, 3, 3, 2, 2, 3, 2, 1, 3, 4, 3, 3],
#             'activity_level': [
#                 'relaxing', 'relaxing', 'active', 'moderate', 'very_active',
#                 'relaxing', 'relaxing', 'moderate', 'moderate', 'relaxing',
#                 'active', 'relaxing', 'relaxing', 'moderate', 'moderate',
#                 'relaxing', 'active', 'relaxing', 'moderate', 'relaxing'
#             ],
#             'popularity': [
#                 'popular', 'popular', 'hidden_gem', 'popular', 'hidden_gem',
#                 'hidden_gem', 'hidden_gem', 'popular', 'very_popular', 'popular',
#                 'popular', 'very_popular', 'hidden_gem', 'popular', 'popular',
#                 'hidden_gem', 'very_popular', 'popular', 'hidden_gem', 'popular'
#             ]
#         }
        
#         # Sample user data
#         users_data = {
#             'user_id': list(range(1, 11)),
#             'preferences': [
#                 'outdoor activities, food, adventure',
#                 'culture, relaxation, art',
#                 'active lifestyle, outdoors, sports',
#                 'entertainment, food, social events',
#                 'wellness, nature, quiet places',
#                 'adventure, nightlife, outdoor activities',
#                 'culture, history, educational experiences',
#                 'food, entertainment, shopping',
#                 'active lifestyle, sports, adventure',
#                 'relaxation, wellness, culture'
#             ]
#         }
        
#         # Sample user ratings
#         ratings_data = []
#         for user_id in range(1, 11):
#             # Each user rates 5-10 places
#             num_ratings = random.randint(5, 10)
#             places_rated = random.sample(range(1, 21), num_ratings)
            
#             for place_id in places_rated:
#                 # Generate a rating between 1 and 5
#                 rating = random.randint(1, 5)
#                 timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
#                 ratings_data.append({
#                     'user_id': user_id,
#                     'place_id': place_id,
#                     'rating': rating,
#                     'timestamp': timestamp
#                 })
        
#         self.places_df = pd.DataFrame(places_data)
#         self.users_df = pd.DataFrame(users_data)
#         self.ratings_df = pd.DataFrame(ratings_data)
        
#         print(f"Loaded sample data: {len(self.places_df)} places, {len(self.users_df)} users, {len(self.ratings_df)} ratings")
#         return True
    
#     def build_content_based_model(self):
#         """Build a content-based recommendation model."""
#         if self.places_df is None:
#             print("Please load data first!")
#             return False
        
#         # Create a combined text feature for TF-IDF
#         self.places_df['combined_features'] = (
#             self.places_df['name'] + ' ' + 
#             self.places_df['category'] + ' ' + 
#             self.places_df['description'] + ' ' + 
#             self.places_df['activity_level'] + ' ' + 
#             self.places_df['popularity']
#         )
        
#         # Create TF-IDF matrix
#         tfidf = TfidfVectorizer(stop_words='english')
#         tfidf_matrix = tfidf.fit_transform(self.places_df['combined_features'])
        
#         # Calculate cosine similarity
#         self.content_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
#         print("Content-based recommendation model built successfully!")
#         return True
    
#     def build_collaborative_filtering_model(self):
#         """Build a collaborative filtering recommendation model."""
#         if self.ratings_df is None:
#             print("Please load data first!")
#             return False
        
#         # Create user-item matrix
#         user_item_df = self.ratings_df.pivot_table(
#             index='user_id', 
#             columns='place_id', 
#             values='rating',
#             fill_value=0
#         )
        
#         # Convert to numpy matrix
#         self.user_item_matrix = user_item_df.values
        
#         # Calculate user similarity (user-based collaborative filtering)
#         self.user_similarity = cosine_similarity(self.user_item_matrix)
        
#         print("Collaborative filtering model built successfully!")
#         return True
    
#     def analyze_mood(self, text):
#         compound_score = self.sia.polarity_scores(text)['compound']
        
#         if compound_score <= -0.5:
#             return 'very_negative'
#         elif -0.5 < compound_score <= -0.1:
#             return 'negative'
#         elif -0.1 < compound_score < 0.1:
#             return 'neutral'
#         elif 0.1 <= compound_score < 0.5:
#             return 'positive'
#         else:
#             return 'very_positive'
    
#     def get_content_recommendations(self, place_id, top_n=5):
#         """Get content-based recommendations based on place similarity."""
#         if self.content_matrix is None:
#             print("Please build the content-based model first!")
#             return []
        
#         idx = self.places_df[self.places_df['place_id'] == place_id].index[0]
        
#         sim_scores = list(enumerate(self.content_matrix[idx]))
#         sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
#         sim_scores = sim_scores[1:top_n+1]
#         place_indices = [i[0] for i in sim_scores]
        
#         return self.places_df.iloc[place_indices]
    
#     def get_user_recommendations(self, user_id, mood_text=None, top_n=5):
#         """Get personalized recommendations for a user, adjusted for mood if provided."""
#         if self.content_matrix is None or self.user_item_matrix is None:
#             print("Please build both recommendation models first!")
#             return []
        
#         if user_id not in self.ratings_df['user_id'].values:
#             print(f"User {user_id} not found in ratings data!")
#             return []
        
#         mood_category = 'neutral'  # Default mood
#         if mood_text:
#             mood_category = self.analyze_mood(mood_text)
#             print(f"Detected mood: {mood_category}")
        
#         mood_weight = self.mood_weights[mood_category]
        
#         # Get places user has already rated
#         user_places = self.ratings_df[self.ratings_df['user_id'] == user_id]['place_id'].values
        
#         # Get weighted place scores
#         place_scores = {}
#         for place_id in self.places_df['place_id'].values:
#             if place_id in user_places:
#                 continue  # Skip already rated places
            
#             place_info = self.places_df[self.places_df['place_id'] == place_id].iloc[0]
            
#             # Calculate base score using content matching
#             base_score = 0
#             for rated_place_id in user_places:
#                 # Get index of rated place
#                 rated_idx = self.places_df[self.places_df['place_id'] == rated_place_id].index[0]
#                 # Get index of candidate place
#                 candidate_idx = self.places_df[self.places_df['place_id'] == place_id].index[0]
#                 # Add similarity score weighted by user's rating
#                 user_rating = self.ratings_df[(self.ratings_df['user_id'] == user_id) & 
#                                              (self.ratings_df['place_id'] == rated_place_id)]['rating'].values[0]
#                 base_score += self.content_matrix[rated_idx][candidate_idx] * user_rating
            
#             # Adjust score based on mood weights
#             activity_multiplier = 1
#             popularity_multiplier = 1
            
#             if place_info['activity_level'] == 'relaxing':
#                 activity_multiplier = mood_weight['relaxing']
#             elif place_info['activity_level'] in ['active', 'very_active']:
#                 activity_multiplier = mood_weight['active']
            
#             if place_info['popularity'] in ['popular', 'very_popular']:
#                 popularity_multiplier = mood_weight['popular']
#             else:
#                 popularity_multiplier = mood_weight['hidden_gem']
            
#             # Apply mood-based adjustment
#             adjusted_score = base_score * activity_multiplier * popularity_multiplier
            
#             place_scores[place_id] = adjusted_score
        
#         # Get top N recommendations
#         top_places = sorted(place_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
#         top_place_ids = [place_id for place_id, _ in top_places]
        
#         recommendations = self.places_df[self.places_df['place_id'].isin(top_place_ids)]
#         return recommendations
    
#     def save_models(self, filename_prefix='smart_recommender'):
#         """Save trained models to disk."""
#         models_data = {
#             'places_df': self.places_df,
#             'users_df': self.users_df,
#             'ratings_df': self.ratings_df,
#             'content_matrix': self.content_matrix,
#             'user_item_matrix': self.user_item_matrix
#         }
        
#         joblib.dump(models_data, f"{filename_prefix}_models.pkl")
#         print(f"Models saved to {filename_prefix}_models.pkl")
    
#     def load_models(self, filename_prefix='smart_recommender'):
#         """Load trained models from disk."""
#         try:
#             models_data = joblib.load(f"{filename_prefix}_models.pkl")
            
#             self.places_df = models_data['places_df']
#             self.users_df = models_data['users_df']
#             self.ratings_df = models_data['ratings_df']
#             self.content_matrix = models_data['content_matrix']
#             self.user_item_matrix = models_data['user_item_matrix']
            
#             print("Models loaded successfully!")
#             return True
#         except Exception as e:
#             print(f"Error loading models: {e}")
#             return False
        

# # Demo application
# def main():
#     print("\n=== Smart Recommendation System with Sentiment Analysis ===\n")
    
#     # Initialize recommender system
#     recommender = SmartRecommender()
    
#     # Load sample data
#     recommender.load_sample_data()
    
#     # Build recommendation models
#     recommender.build_content_based_model()
#     recommender.build_collaborative_filtering_model()
    
#     # Test recommendation system with different moods
#     test_user_id = 3
    
#     print("\n--- Content-based recommendations for Green Garden Restaurant ---")
#     similar_places = recommender.get_content_recommendations(place_id=1, top_n=3)
#     print(similar_places[['name', 'category', 'activity_level']])
    
#     print("\n--- Personalized recommendations with different moods ---")
    
#     print("\n1. With happy mood:")
#     happy_mood = "I'm feeling fantastic today! Everything is wonderful!"
#     happy_recommendations = recommender.get_user_recommendations(
#         user_id=test_user_id, 
#         mood_text=happy_mood, 
#         top_n=3
#     )
#     print(happy_recommendations[['name', 'category', 'activity_level']])
    
#     print("\n2. With sad mood:")
#     sad_mood = "I'm feeling down today. Nothing seems to be going right."
#     sad_recommendations = recommender.get_user_recommendations(
#         user_id=test_user_id, 
#         mood_text=sad_mood, 
#         top_n=3
#     )
#     print(sad_recommendations[['name', 'category', 'activity_level']])
    
#     print("\n3. With neutral mood:")
#     neutral_mood = "It's just another day."
#     neutral_recommendations = recommender.get_user_recommendations(
#         user_id=test_user_id, 
#         mood_text=neutral_mood, 
#         top_n=3
#     )
#     print(neutral_recommendations[['name', 'category', 'activity_level']])
    
#     # Save models
#     recommender.save_models()
    
#     print("\n=== Demo Complete ===")

# if __name__ == "__main__":
#     main()

import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import joblib
from datetime import datetime
import random

# Download NLTK resources for sentiment analysis
nltk.download('vader_lexicon', quiet=True)

class SmartRecommender:
    def __init__(self):
        """Initialize the Smart Recommendation System."""
        self.places_df = None
        self.users_df = None
        self.ratings_df = None
        self.content_matrix = None
        self.user_item_matrix = None
        self.sia = SentimentIntensityAnalyzer()
        self.mood_weights = {
            'very_negative': {'active': 0.1, 'relaxing': 0.9, 'popular': 0.3, 'hidden_gem': 0.7},
            'negative': {'active': 0.3, 'relaxing': 0.7, 'popular': 0.4, 'hidden_gem': 0.6},
            'neutral': {'active': 0.5, 'relaxing': 0.5, 'popular': 0.5, 'hidden_gem': 0.5},
            'positive': {'active': 0.7, 'relaxing': 0.3, 'popular': 0.6, 'hidden_gem': 0.4},
            'very_positive': {'active': 0.9, 'relaxing': 0.1, 'popular': 0.7, 'hidden_gem': 0.3}
        }
        
    def load_sample_data(self):
        """Load sample data for demonstration purposes."""
        # Sample data for places (restaurants, activities, etc.)
        places_data = {
            'place_id': list(range(1, 21)),
            'name': [
                'Green Garden Restaurant', 'Sunset Beach', 'Mountain Trail Hike',
                'City Museum', 'Sky Diving Experience', 'Cozy Cafe', 
                'Historic Library', 'Botanical Gardens', 'Central Park',
                'Harbor Cruise', 'Rock Climbing Center', 'Downtown Cinema',
                'Art Gallery', 'Yoga Studio', 'Live Music Bar',
                'Lakeside Picnic Area', 'Adventure Park', 'Spa Retreat',
                'Local Brewery Tour', 'Theatre Performance'
            ],
            'category': [
                'restaurant', 'outdoor', 'outdoor', 'culture', 'adventure',
                'restaurant', 'culture', 'outdoor', 'outdoor', 'adventure',
                'adventure', 'entertainment', 'culture', 'wellness', 'entertainment',
                'outdoor', 'adventure', 'wellness', 'entertainment', 'culture'
            ],
            'description': [
                'Organic vegetarian restaurant with garden seating',
                'Beautiful beach with stunning sunset views',
                'Challenging mountain trail with scenic overlooks',
                'Expansive museum showcasing city history and art',
                'Thrilling sky diving experience for adrenaline seekers',
                'Intimate cafe with specialty coffees and pastries',
                'Historic library with rare book collection',
                'Extensive gardens featuring exotic plants and flowers',
                'Large urban park with walking paths and ponds',
                'Relaxing cruise around the harbor with city views',
                'Indoor climbing center with routes for all levels',
                'Modern cinema showing latest blockbusters',
                'Contemporary art gallery with rotating exhibits',
                'Peaceful yoga studio offering various classes',
                'Energetic bar featuring live local bands',
                'Serene picnic area by the lake with BBQ facilities',
                'Family adventure park with ziplines and obstacle courses',
                'Luxury spa offering massage and wellness treatments',
                'Guided tour of local craft breweries with tastings',
                'Theater showing acclaimed performances and plays'
            ],
            'price_level': [3, 1, 2, 2, 4, 2, 1, 2, 1, 3, 3, 2, 2, 3, 2, 1, 3, 4, 3, 3],
            'activity_level': [
                'relaxing', 'relaxing', 'active', 'moderate', 'very_active',
                'relaxing', 'relaxing', 'moderate', 'moderate', 'relaxing',
                'active', 'relaxing', 'relaxing', 'moderate', 'moderate',
                'relaxing', 'active', 'relaxing', 'moderate', 'relaxing'
            ],
            'popularity': [
                'popular', 'popular', 'hidden_gem', 'popular', 'hidden_gem',
                'hidden_gem', 'hidden_gem', 'popular', 'very_popular', 'popular',
                'popular', 'very_popular', 'hidden_gem', 'popular', 'popular',
                'hidden_gem', 'very_popular', 'popular', 'hidden_gem', 'popular'
            ]
        }
        
        # Sample user data
        users_data = {
            'user_id': list(range(1, 11)),
            'preferences': [
                'outdoor activities, food, adventure',
                'culture, relaxation, art',
                'active lifestyle, outdoors, sports',
                'entertainment, food, social events',
                'wellness, nature, quiet places',
                'adventure, nightlife, outdoor activities',
                'culture, history, educational experiences',
                'food, entertainment, shopping',
                'active lifestyle, sports, adventure',
                'relaxation, wellness, culture'
            ]
        }
        
        # Sample user ratings
        ratings_data = []
        for user_id in range(1, 11):
            # Each user rates 5-10 places
            num_ratings = random.randint(5, 10)
            places_rated = random.sample(range(1, 21), num_ratings)
            
            for place_id in places_rated:
                # Generate a rating between 1 and 5
                rating = random.randint(1, 5)
                timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                ratings_data.append({
                    'user_id': user_id,
                    'place_id': place_id,
                    'rating': rating,
                    'timestamp': timestamp
                })
        
        self.places_df = pd.DataFrame(places_data)
        self.users_df = pd.DataFrame(users_data)
        self.ratings_df = pd.DataFrame(ratings_data)
        
        print(f"Loaded sample data: {len(self.places_df)} places, {len(self.users_df)} users, {len(self.ratings_df)} ratings")
        return True
    
    def build_content_based_model(self):
        """Build a content-based recommendation model."""
        if self.places_df is None:
            print("Please load data first!")
            return False
        
        # Create a combined text feature for TF-IDF
        self.places_df['combined_features'] = (
            self.places_df['name'] + ' ' + 
            self.places_df['category'] + ' ' + 
            self.places_df['description'] + ' ' + 
            self.places_df['activity_level'] + ' ' + 
            self.places_df['popularity']
        )
        
        # Create TF-IDF matrix
        tfidf = TfidfVectorizer(stop_words='english')
        tfidf_matrix = tfidf.fit_transform(self.places_df['combined_features'])
        
        # Calculate cosine similarity
        self.content_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        
        print("Content-based recommendation model built successfully!")
        return True
    
    def build_collaborative_filtering_model(self):
        """Build a collaborative filtering recommendation model."""
        if self.ratings_df is None:
            print("Please load data first!")
            return False
        
        # Create user-item matrix
        user_item_df = self.ratings_df.pivot_table(
            index='user_id', 
            columns='place_id', 
            values='rating',
            fill_value=0
        )
        
        # Convert to numpy matrix
        self.user_item_matrix = user_item_df.values
        
        # Calculate user similarity (user-based collaborative filtering)
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        
        print("Collaborative filtering model built successfully!")
        return True
    
    def analyze_mood(self, text):
        compound_score = self.sia.polarity_scores(text)['compound']
        
        if compound_score <= -0.5:
            return 'very_negative'
        elif -0.5 < compound_score <= -0.1:
            return 'negative'
        elif -0.1 < compound_score < 0.1:
            return 'neutral'
        elif 0.1 <= compound_score < 0.5:
            return 'positive'
        else:
            return 'very_positive'
    
    def get_content_recommendations(self, place_id, top_n=5):
        """Get content-based recommendations based on place similarity."""
        if self.content_matrix is None:
            print("Please build the content-based model first!")
            return []
        
        idx = self.places_df[self.places_df['place_id'] == place_id].index[0]
        
        sim_scores = list(enumerate(self.content_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        
        sim_scores = sim_scores[1:top_n+1]
        place_indices = [i[0] for i in sim_scores]
        
        return self.places_df.iloc[place_indices]
    
    def get_user_recommendations(self, user_id, mood_text=None, top_n=5):
        """Get personalized recommendations for a user, adjusted for mood if provided."""
        if self.content_matrix is None or self.user_item_matrix is None:
            print("Please build both recommendation models first!")
            return []
        
        if user_id not in self.ratings_df['user_id'].values:
            print(f"User {user_id} not found in ratings data!")
            return []
        
        mood_category = 'neutral'  # Default mood
        if mood_text:
            mood_category = self.analyze_mood(mood_text)
            print(f"Detected mood: {mood_category}")
        
        mood_weight = self.mood_weights[mood_category]
        
        # Get places user has already rated
        user_places = self.ratings_df[self.ratings_df['user_id'] == user_id]['place_id'].values
        
        # Get weighted place scores
        place_scores = {}
        for place_id in self.places_df['place_id'].values:
            if place_id in user_places:
                continue  # Skip already rated places
            
            place_info = self.places_df[self.places_df['place_id'] == place_id].iloc[0]
            
            # Calculate base score using content matching
            base_score = 0
            for rated_place_id in user_places:
                # Get index of rated place
                rated_idx = self.places_df[self.places_df['place_id'] == rated_place_id].index[0]
                # Get index of candidate place
                candidate_idx = self.places_df[self.places_df['place_id'] == place_id].index[0]
                # Add similarity score weighted by user's rating
                user_rating = self.ratings_df[(self.ratings_df['user_id'] == user_id) & 
                                             (self.ratings_df['place_id'] == rated_place_id)]['rating'].values[0]
                base_score += self.content_matrix[rated_idx][candidate_idx] * user_rating
            
            # Adjust score based on mood weights
            activity_multiplier = 1
            popularity_multiplier = 1
            
            if place_info['activity_level'] == 'relaxing':
                activity_multiplier = mood_weight['relaxing']
            elif place_info['activity_level'] in ['active', 'very_active']:
                activity_multiplier = mood_weight['active']
            
            if place_info['popularity'] in ['popular', 'very_popular']:
                popularity_multiplier = mood_weight['popular']
            else:
                popularity_multiplier = mood_weight['hidden_gem']
            
            # Apply mood-based adjustment
            adjusted_score = base_score * activity_multiplier * popularity_multiplier
            
            place_scores[place_id] = adjusted_score
        
        # Get top N recommendations
        top_places = sorted(place_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
        top_place_ids = [place_id for place_id, _ in top_places]
        
        recommendations = self.places_df[self.places_df['place_id'].isin(top_place_ids)]
        return recommendations
    
    def save_models(self, filename_prefix='smart_recommender'):
        """Save trained models to disk."""
        models_data = {
            'places_df': self.places_df,
            'users_df': self.users_df,
            'ratings_df': self.ratings_df,
            'content_matrix': self.content_matrix,
            'user_item_matrix': self.user_item_matrix
        }
        
        joblib.dump(models_data, f"{filename_prefix}_models.pkl")
        print(f"Models saved to {filename_prefix}_models.pkl")
    
    def load_models(self, filename_prefix='smart_recommender'):
        """Load trained models from disk."""
        try:
            models_data = joblib.load(f"{filename_prefix}_models.pkl")
            
            self.places_df = models_data['places_df']
            self.users_df = models_data['users_df']
            self.ratings_df = models_data['ratings_df']
            self.content_matrix = models_data['content_matrix']
            self.user_item_matrix = models_data['user_item_matrix']
            
            print("Models loaded successfully!")
            return True
        except Exception as e:
            print(f"Error loading models: {e}")
            return False
        
    def evaluate_model(self):
        """Evaluate the model's effectiveness with metrics."""
        if self.ratings_df is None or self.user_item_matrix is None:
            print("Please load data and build collaborative model first.")
            return
        
        actual = []
        predicted = []

        for user_id in self.ratings_df['user_id'].unique():
            user_ratings = self.ratings_df[self.ratings_df['user_id'] == user_id]
            for _, row in user_ratings.iterrows():
                place_id = row['place_id']
                rating = row['rating']
                recs = self.get_user_recommendations(user_id, top_n=10)
                if place_id in recs['place_id'].values:
                    predicted.append(5)  # Assume highly recommended
                else:
                    predicted.append(0)  # Not recommended
                actual.append(rating)

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)
        print(f"\nModel Evaluation:\nRMSE: {rmse:.2f}\nMAE: {mae:.2f}")
        

# Demo application
def main():
    print("\n=== Smart Recommendation System with Sentiment Analysis ===\n")
    
    # Initialize recommender system
    recommender = SmartRecommender()
    
    # Load sample data
    recommender.load_sample_data()
    
    # Build recommendation models
    recommender.build_content_based_model()
    recommender.build_collaborative_filtering_model()
    
    # Test recommendation system with different moods
    test_user_id = 3
    
    print("\n--- Content-based recommendations for Green Garden Restaurant ---")
    similar_places = recommender.get_content_recommendations(place_id=1, top_n=3)
    print(similar_places[['name', 'category', 'activity_level']])
    
    print("\n--- Personalized recommendations with different moods ---")
    
    print("\n1. With happy mood:")
    happy_mood = "I'm feeling fantastic today! Everything is wonderful!"
    happy_recommendations = recommender.get_user_recommendations(
        user_id=test_user_id, 
        mood_text=happy_mood, 
        top_n=3
    )
    print(happy_recommendations[['name', 'category', 'activity_level']])
    
    print("\n2. With sad mood:")
    sad_mood = "I'm feeling down today. Nothing seems to be going right."
    sad_recommendations = recommender.get_user_recommendations(
        user_id=test_user_id, 
        mood_text=sad_mood, 
        top_n=3
    )
    print(sad_recommendations[['name', 'category', 'activity_level']])
    
    print("\n3. With neutral mood:")
    neutral_mood = "It's just another day."
    neutral_recommendations = recommender.get_user_recommendations(
        user_id=test_user_id, 
        mood_text=neutral_mood, 
        top_n=3
    )
    print(neutral_recommendations[['name', 'category', 'activity_level']])
    
    # Save models
    recommender.save_models()
    
    print("\n=== Demo Complete ===")

if __name__ == "__main__":
    main()
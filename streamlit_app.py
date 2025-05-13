import streamlit as st
import pandas as pd
from smart_recommender import SmartRecommender
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Initialize recommender system
recommender = SmartRecommender()
recommender.load_sample_data()
recommender.build_content_based_model()
recommender.build_collaborative_filtering_model()

# Sidebar for Navigation
st.sidebar.title("Smart Recommender")
st.sidebar.markdown("""
    Choose an option from the left. This app recommends places based on your mood and preferences.
""")

# Tabs for different sections
tabs = st.sidebar.radio("Select Tab", ["Home", "Recommendations", "Similar Places", "Data", "Evaluation"])


if tabs == "Home":
    # Home Page Content
    st.title("Welcome to Smart Recommender!")
    st.subheader("AI-based recommendation system powered Streamlit.")
    
    st.markdown("""
        ### How it works:
        1. **User Input**: You provide your user ID, mood, and the number of recommendations you want.
        2. **AI Recommendations**: The system returns personalized recommendations based on your preferences.
        3. **Similar Places**: You can also explore similar places based on the one you like!
    """)

elif tabs == "Recommendations":
    # Recommendations Section
    st.title("Get Recommendations")
    
    # User input for recommendations
    user_id = st.selectbox("Select User ID", recommender.users_df['user_id'].unique())
    mood_text = st.text_input("What mood are you in?", "Happy")
    top_n = st.slider("Number of Recommendations", min_value=1, max_value=10, value=5)

    # Button to submit
    if st.button("Get Recommendations"):
        recommendations = recommender.get_user_recommendations(user_id=user_id, mood_text=mood_text, top_n=top_n)
        st.subheader(f"Recommendations for User {user_id} with mood {mood_text}")

        # Display recommendations
        if not recommendations.empty:
            for _, rec in recommendations.iterrows():
                st.markdown(f"""
                    **{rec['name']}**  
                    Category: {rec['category']}  
                    Description: {rec['description']}  
                    Activity Level: {rec['activity_level']}  
                    Popularity: {rec['popularity']}  
                """)
        else:
            st.write("No recommendations found.")

elif tabs == "Similar Places":
    # Similar Places Section
    st.title("Find Similar Places")

    # User input for similar places
    place_id = st.selectbox("Select Place ID", recommender.places_df['place_id'].unique())
    top_n = st.slider("Number of Similar Places", min_value=1, max_value=10, value=5)

    # Button to submit
    if st.button("Get Similar Places"):
        similar_places = recommender.get_content_recommendations(place_id=place_id, top_n=top_n)
        st.subheader(f"Similar Places to Place ID {place_id}")
        
        # Display similar places
        if not similar_places.empty:
            for _, place in similar_places.iterrows():
                st.markdown(f"""
                    **{place['name']}**  
                    Category: {place['category']}  
                    Description: {place['description']}  
                """)
        else:
            st.write("No similar places found.")

elif tabs == "Data":
    # Data Section
    st.title("Available Data")

    # Displaying Data
    st.subheader("Places Data")
    st.write(recommender.places_df)

    st.subheader("Users Data")
    st.write(recommender.users_df)

elif tabs == "Evaluation":
    st.title("Model Evaluation Metrics")

    # Run evaluation
    with st.spinner("Evaluating model..."):
        actual = []
        predicted = []

        for user_id in recommender.ratings_df['user_id'].unique():
            user_ratings = recommender.ratings_df[recommender.ratings_df['user_id'] == user_id]

            for _, row in user_ratings.iterrows():
                place_id = row['place_id']
                rating = row['rating']
                actual.append(rating)

                # Predict rating using content similarity
                rated_places = user_ratings[user_ratings['place_id'] != place_id]
                sim_sum = 0
                weighted_sum = 0

                for _, rated_row in rated_places.iterrows():
                    i = recommender.places_df[recommender.places_df['place_id'] == place_id].index[0]
                    j = recommender.places_df[recommender.places_df['place_id'] == rated_row['place_id']].index[0]
                    sim = recommender.content_matrix[i][j]
                    weighted_sum += sim * rated_row['rating']
                    sim_sum += sim

                if sim_sum > 0:
                    pred_rating = weighted_sum / sim_sum
                else:
                    pred_rating = 3.0  # fallback neutral rating

                predicted.append(pred_rating)

        rmse = np.sqrt(mean_squared_error(actual, predicted))
        mae = mean_absolute_error(actual, predicted)

        st.metric("RMSE (Root Mean Squared Error)", f"{rmse:.2f}")
        st.metric("MAE (Mean Absolute Error)", f"{mae:.2f}")

        # Bar chart
        chart_data = pd.DataFrame({
            "Actual Ratings": actual[:20],
            "Predicted Ratings": predicted[:20]
        })
        st.subheader("Sample Rating Comparison (First 20)")
        st.bar_chart(chart_data)


# Smart Recommendation System

- A robust and scalable recommendation system designed to offer personalized suggestions based on both content-based and collaborative filtering models. This system employs multiple machine learning techniques to provide high-quality recommendations to users. 

- [Live App](https://imabhnv-smart-recommendation-system.streamlit.app/)

## Overview

The **Smart Recommendation System** leverages:
1. **Content-based Filtering** - Recommends items similar to those the user has shown interest in, based on item features.
2. **Collaborative Filtering** - Recommends items based on user-item interactions, utilizing user preferences and behavior.

Additionally, the system is integrated with an **evaluation module** to assess model performance using accuracy metrics.

## Features

- **Hybrid Recommendation**: Combines content-based and collaborative filtering methods.
- **Scalable Model**: Suitable for large datasets with millions of user-item interactions.
- **Model Evaluation**: Built-in model evaluation functions for performance assessment.
- **Extensibility**: Easy to add more recommendation models or data processing steps.

## Project Structure

```plaintext
Smart-Recommendation-System/
│
├── app.py                    # Main execution file to run the recommendation system.
├── requirements.txt          # List of dependencies required to run the project.
├── smart_recommender.py      # Core logic for building recommendation models.
└── README.md                 # This README file.
```

## Installation

Follow these steps to set up the environment:

1. Clone this repository:

   ```bash
   git clone https://github.com/imabhnv/Smart-Recommendation-System.git
   ```

2. Navigate into the project directory:

   ```bash
   cd Smart-Recommendation-System
   ```

3. Install required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. **Load Data**: First of all run the smart_recommender.py file to generate model file.

2. **Run the Recommendation System**:

   The `app.py` file runs the system and produces recommendations based on the provided dataset.

   ```bash
   streamlit run streamlit_app.py
   ```

## Evaluation

- **Content-based model**: Evaluates based on item similarity and features.
- **Collaborative filtering model**: Evaluates based on user-item interaction patterns.

## Dependencies

The following Python packages are required for the system to run:

- pandas
- numpy
- scikit-learn
- scipy
- matplotlib
- seaborn

These dependencies are listed in the `requirements.txt` file.

## Contributing

If you'd like to contribute to this project, feel free to fork the repository, make changes, and submit a pull request. Please make sure to follow the coding standards and test your changes thoroughly.

## Made with 💻♥️ by Abhinav Varshney🚀

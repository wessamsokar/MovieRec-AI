# CET251 - AI: Movie Recommendation & Viewing Planner

**Course:** CET251 - Introduction to Artificial Intelligence  
**Institution:** Elsewedy Polytechnic University

## Project Overview

This project is a fully-functional Movie Recommendation and Viewing Planner prototype that leverages Artificial Intelligence and Machine Learning to help users discover films based on their specific tastes. It implements search and prediction techniques to accurately rank and explain personalized movie recommendations.

## Core AI Concepts Used

1. **Content-Based Recommendation (TF-IDF & Cosine Similarity):** 
   - Builds a text vector representation from movie genres and overviews. 
   - Computes cosine similarity to recommend movies contextually similar to the user's selected favorites.
2. **Heuristic/Rule-Based Recommendation:** 
   - Ranks movies using a robust weighted heuristic (70% Average Rating, 30% Popularity).
   - Dynamically filters based on strict user constraints (Runtime, Adult Content, Genres).
3. **Neural Network Component (MLPClassifier):** 
   - Trains a Multi-Layer Perceptron (MLP) on the user's rating history.
   - Predicts a personalized "Preference Score" indicating the likelihood a user will enjoy a recommended movie based on features like runtime, vote average, release year, and popularity.

## File Structure

- `app.py`: The main Streamlit web application, containing UI elements, recommendation logic (Content-Based and Heuristic), and Neural Network inference.
- `data_loader.py`: Handles data ingestion, cleaning, and strict subsetting to maintain the required dataset size constraint of 100-500 movies for optimal performance.
- `movies_metadata.csv`: Contains the dataset of movie metadata used for features, TF-IDF vectorization, and UI display.
- `ratings_small.csv`: Contains user interactions and ratings used to train the Neural Network.
- `requirements.txt`: Specifies the exact Python library dependencies to ensure seamless cross-platform execution.
- `.gitignore`: Standard exclusion list for repository hygiene.

## Prerequisites and Installation

1. Clone or download this project workspace.
2. Ensure you have Python installed.
3. Install the required dependencies using `pip`:

```bash
pip install -r requirements.txt
```

## How to Run the App

Execute the following command in your terminal from the project's root directory:

```bash
python -m streamlit run app.py
```
*(Or use `py -m streamlit run app.py` on Windows)*

Once the local server starts, the app will automatically open in your default web browser.

## Using the Interface

1. **Set Preferences:** In the left sidebar, select up to 3 movies you like and optionally choose preferred genres.
2. **Set Constraints:** Adjust the Max Runtime slider and toggle the Family Friendly checkbox based on your viewing availability.
3. **Generate:** Click the **Generate Recommendations** button.
4. **Evaluate:** The application will display Method A (Content-Based) and Method B (Heuristic) recommendations side-by-side, complete with textual explanations and Neural Network preference predictions, followed by an explicit Method Comparison summary.

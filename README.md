# CET251 - AI: Movie Recommendation & Viewing Planner

**Course:** CET251 - Introduction to Artificial Intelligence  
**Institution:** Elsewedy Polytechnic University

## Project Overview

This project is a Movie Recommendation and Viewing Planner prototype that leverages Artificial Intelligence and Machine Learning for discovery and ranking. Content-based and KNN tracks use explicit user seeds where applicable; the MLP provides a **catalog-level** appeal signal trained on pooled ratings—not user-specific personalization.

## Core AI Concepts Used

1. **Content-Based Recommendation (TF-IDF & Cosine Similarity):**
   - Builds a text vector representation from movie genres and overviews.
   - Computes cosine similarity to movies **seeded by liked titles** chosen in the UI (Method A requires these anchors).

2. **Heuristic / Rule-Based Recommendation:**
   - Ranks candidates with a **Normalized Weighted Score: 0.7×Rating + 0.3×Popularity**, where Rating and Popularity are each divided by the **maximum value in the current filtered subset** before weighting (not raw linear units).
   - Applies strict user constraints (runtime, adult content, genres, and optional era in Cold-Start mode).

3. **Item-Based Nearest Neighbors (Metadata KNN — Method C):**
   - Trains `NearestNeighbors` on **movie feature vectors** (genre TF-IDF plus scaled metadata such as rating, popularity, runtime, year)—**not** a user–item ratings matrix, so this is metadata similarity rather than classical collaborative filtering.

4. **Neural Network — Global Taste Model (MLPClassifier):**
   - Trains a single **global** Multi-Layer Perceptron on the Kaggle-derived **`ratings_small.csv`** subset (**100k+ rating rows**) merged with movie features.
   - Predicts **general movie appeal** (like vs. dislike from pooled users) from features such as runtime, vote average, release year, and popularity—the **same** score for every app user for a given movie, not a separately trained **per-user** model.

## 🌟 Bonus Features (Stretch Goals)

This prototype includes several advanced features that go beyond the core requirements to enhance the user experience:

- **🧊 Cold-Start User Mode:** Asks three questions (story vibe, time available, era) to infer genres and filters. **Method A (content-based)** and **Method C (item KNN)** are **intentionally skipped** without liked movies; **Method B (heuristic)** drives discovery in this mode by design.
- **🎭 Mood-Based Recommendations:** Instead of manually picking genres, users can select their current emotional state (e.g., Happy, Adventurous, Thoughtful). The system dynamically maps this mood to the optimal set of movie genres.
- **🎨 Genre Diversity Control:** A dynamic slider (0% to 100%) that enforces content variety. When set higher, the filtering algorithm actively limits how many recommendations can share the same primary genre, ensuring a well-rounded list.
- **📅 Watch Plan:** Builds a short day-by-day schedule from combined recommendation pools, ordered by the **global taste MLP** score (then genre-diversity rules), not per-user neural personalization.

## File Structure

- `app.py`: The main Streamlit web application (content-based, heuristic, item-based KNN, and **global** MLP inference).
- `data_loader.py`: Handles data ingestion, cleaning, and strict subsetting to maintain the required dataset size constraint of 100-500 movies for optimal performance.
- `movies_metadata.csv`: Contains the dataset of movie metadata used for features, TF-IDF vectorization, and UI display.
- `ratings_small.csv`: Kaggle MovieLens-style ratings (100k+ rows) used to train the **global taste** MLP on pooled user–movie interactions.
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

_(Or use `py -m streamlit run app.py` on Windows)_

Once the local server starts, the app will automatically open in your default web browser.

## Using the Interface

1. **Set Preferences:** In the left sidebar, select up to 3 movies you like and optionally choose preferred genres.
2. **Set Constraints:** Adjust the Max Runtime slider and toggle the Family Friendly checkbox based on your viewing availability.
3. **Generate:** Click the **Generate Recommendations** button.
4. **Evaluate:** The application shows Method A (content-based), Method B (normalized heuristic score), and Method C (item-based KNN) where seeds exist, plus **global MLP** appeal scores and a method comparison summary.

## Dataset

This project uses the https://www.kaggle.com/datasets/rounakbanik/the-movies-dataset.
Only used 2 files Metadata and Rating

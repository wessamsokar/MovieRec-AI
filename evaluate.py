import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from data_loader import load_movies, load_ratings


def build_content_similarity(movies_df):
    content = movies_df["genres"].fillna("") + " " + movies_df["overview"].fillna("")
    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(content)
    return cosine_similarity(tfidf_matrix, tfidf_matrix)


def build_knn(movies_df):
    genre_tfidf = TfidfVectorizer(stop_words="english")
    genre_matrix = genre_tfidf.fit_transform(movies_df["genres"].fillna(""))

    numeric_cols = ["vote_average", "popularity", "runtime", "year"]
    numeric = movies_df[numeric_cols].fillna(0).to_numpy(dtype=float)
    scaler = StandardScaler()
    numeric_scaled = scaler.fit_transform(numeric)

    features = np.hstack([genre_matrix.toarray(), numeric_scaled])
    n_neighbors = min(21, len(movies_df))
    knn = NearestNeighbors(metric="cosine", algorithm="brute", n_neighbors=n_neighbors)
    knn.fit(features)
    return knn, features


def get_content_recommendations(seed_index, similarity_matrix, k):
    scores = similarity_matrix[seed_index]
    ranked = np.argsort(scores)[::-1]
    ranked = [i for i in ranked if i != seed_index]
    return ranked[:k]


def get_knn_recommendations(seed_index, knn_model, feature_matrix, k):
    distances, indices = knn_model.kneighbors(
        feature_matrix[seed_index].reshape(1, -1),
        n_neighbors=min(k + 1, len(feature_matrix)),
    )
    ranked = []
    for idx in indices[0]:
        if idx == seed_index:
            continue
        ranked.append(idx)
        if len(ranked) >= k:
            break
    return ranked


def precision_recall_at_k(recommended, relevant_set, k):
    if k == 0:
        return 0.0, 0.0
    hits = len(set(recommended[:k]).intersection(relevant_set))
    precision = hits / k
    recall = hits / len(relevant_set) if relevant_set else 0.0
    return precision, recall


def average_pairwise_diversity(recommended_indices, similarity_matrix):
    if len(recommended_indices) < 2:
        return 0.0
    distances = []
    for i in range(len(recommended_indices)):
        for j in range(i + 1, len(recommended_indices)):
            sim = similarity_matrix[recommended_indices[i], recommended_indices[j]]
            distances.append(1.0 - sim)
    return float(np.mean(distances)) if distances else 0.0


def run_evaluation(k=5):
    movies_df = load_movies("movies_metadata.csv", min_movies=100, max_movies=500)
    ratings_df = load_ratings("ratings_small.csv")

    merged = pd.merge(ratings_df, movies_df, left_on="movieId", right_on="id", how="inner")
    if len(merged) < 10:
        print("Not enough merged data for evaluation.")
        return

    # Build per-user positive interactions.
    merged["liked"] = (merged["rating"] >= 4.0).astype(int)
    positives = merged[merged["liked"] == 1].copy()

    # Keep users with enough positives for a small train/test split.
    user_counts = positives.groupby("userId")["movieId"].nunique()
    valid_users = user_counts[user_counts >= 3].index
    positives = positives[positives["userId"].isin(valid_users)]
    if positives.empty:
        print("No users with enough positive interactions for evaluation.")
        return

    sim_matrix = build_content_similarity(movies_df)
    knn_model, knn_features = build_knn(movies_df)
    id_to_index = {int(row["id"]): idx for idx, row in movies_df.reset_index(drop=True).iterrows()}

    precisions = []
    recalls = []
    diversity_scores = []
    catalog_recommended = set()
    evaluated_users = 0

    for user_id, group in positives.groupby("userId"):
        movie_ids = [int(m) for m in group["movieId"].unique() if int(m) in id_to_index]
        if len(movie_ids) < 3:
            continue

        train_ids, test_ids = train_test_split(movie_ids, test_size=0.33, random_state=42)
        if not train_ids or not test_ids:
            continue

        seed_index = id_to_index[train_ids[0]]
        relevant_indices = {id_to_index[m] for m in test_ids if m in id_to_index}
        if not relevant_indices:
            continue

        # Hybrid list: half content-based + half KNN collaborative-style.
        content_recs = get_content_recommendations(seed_index, sim_matrix, k)
        knn_recs = get_knn_recommendations(seed_index, knn_model, knn_features, k)
        combined = []
        for idx in content_recs + knn_recs:
            if idx != seed_index and idx not in combined:
                combined.append(idx)
            if len(combined) >= k:
                break
        if len(combined) < k:
            continue

        p, r = precision_recall_at_k(combined, relevant_indices, k)
        precisions.append(p)
        recalls.append(r)
        diversity_scores.append(average_pairwise_diversity(combined, sim_matrix))
        catalog_recommended.update(combined)
        evaluated_users += 1

    if evaluated_users == 0:
        print("No valid users were evaluated. Try relaxing thresholds.")
        return

    coverage = len(catalog_recommended) / len(movies_df) if len(movies_df) else 0.0

    print("=" * 60)
    print("Movie Recommender Evaluation Report")
    print("=" * 60)
    print(f"Users Evaluated          : {evaluated_users}")
    print(f"Catalog Size             : {len(movies_df)}")
    print(f"K                        : {k}")
    print("-" * 60)
    print(f"Precision@{k:<2}           : {np.mean(precisions):.4f}")
    print(f"Recall@{k:<2}              : {np.mean(recalls):.4f}")
    print(f"Coverage                 : {coverage:.4f} ({coverage * 100:.2f}%)")
    print(f"Diversity                : {np.mean(diversity_scores):.4f}")
    print("=" * 60)


if __name__ == "__main__":
    run_evaluation(k=5)

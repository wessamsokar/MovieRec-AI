
import pandas as pd
import ast


def load_movies(path="data/movies_metadata.csv", min_votes=50, min_movies=100, max_movies=500):
    df = pd.read_csv(path, low_memory=False)

    def parse_genres(x):
        try:
            return ",".join([g["name"] for g in ast.literal_eval(x) if "name" in g])  
        except Exception:
            return ""

    df["genres"] = df["genres"].apply(parse_genres)

    for col in ["vote_average", "vote_count", "runtime", "popularity"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["year"]  = pd.to_datetime(df["release_date"], errors="coerce").dt.year
    df["id"]    = pd.to_numeric(df["id"], errors="coerce")
    df["adult"] = df["adult"].astype(str).str.lower() == "true"

    df = df[["id","title","genres","runtime","vote_average",
             "vote_count","year","popularity","overview","adult","imdb_id"]]
    df = df.drop_duplicates(subset=["title"])

    # Strict filter first
    filtered = df[
        (df["vote_count"] >= min_votes)
        & (df["vote_average"] > 0)
        & (df["genres"] != "")
        & df["title"].notna()
        & df["runtime"].notna()
        & (df["runtime"] > 0)
        & df["year"].notna()
        & df["id"].notna()
    ].copy()

    # If strict filtering yields too few movies, relax vote_count to reach min_movies
    if len(filtered) < min_movies:
        relaxed = df[
            (df["vote_average"] > 0)
            & (df["genres"] != "")
            & df["title"].notna()
            & df["runtime"].notna()
            & (df["runtime"] > 0)
            & df["year"].notna()
            & df["id"].notna()
        ].copy()
        relaxed = relaxed.sort_values(by="vote_count", ascending=False)
        needed = min_movies - len(filtered)
        extra = relaxed[~relaxed["title"].isin(filtered["title"])].head(needed)
        filtered = pd.concat([filtered, extra], ignore_index=True)

    filtered = filtered.head(max_movies).reset_index(drop=True)
    if len(filtered) < min_movies:
        print(f"[DataLoader] Warning: only {len(filtered)} movies available after filtering.")

    print(f"[DataLoader] {len(filtered)} movies loaded.")
    return filtered


def load_ratings(path="data/ratings_small.csv"):
    df = pd.read_csv(path)
    for col in ["movieId","userId","rating"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna()
    print(f"[DataLoader] {len(df)} ratings from {df['userId'].nunique()} users.")
    return df


def get_genre_list(movies_df):
    genres = set()
    for row in movies_df["genres"]:
        for g in str(row).split(","):
            g = g.strip()
            if g:
                genres.add(g)
    return sorted(genres)
if __name__ == "__main__":
    movies = load_movies()

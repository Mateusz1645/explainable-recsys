"""Movie-to-movie and user-profile recommendation functions."""
import numpy as np
import pandas as pd


def build_title_to_index(movie_profiles: pd.DataFrame) -> pd.Series:
    """Map movie title -> row index, keeping only the first occurrence per title."""
    s = pd.Series(movie_profiles.index.values, index=movie_profiles["title"])
    return s[~s.index.duplicated(keep="first")]


def get_recommendations(
    movie_title: str,
    movie_profiles: pd.DataFrame,
    embeddings: np.ndarray,
    title_to_index: pd.Series,
    n: int = 10,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Top-N movies most similar to ``movie_title`` by embedding cosine similarity."""
    if movie_title not in title_to_index:
        if verbose:
            matches = [t for t in title_to_index.index if movie_title.lower() in t.lower()]
            if matches:
                print(f"'{movie_title}' not found. Did you mean:")
                for m in matches[:5]:
                    print(f"  -> {m}")
            else:
                print(f"Movie '{movie_title}' not found.")
        return None

    idx = int(title_to_index[movie_title])
    scores = embeddings @ embeddings[idx]
    top_n = np.argsort(scores)[::-1][1 : n + 1]

    result = movie_profiles[["title", "year", "genres_str"]].iloc[top_n].copy()
    result["similarity"] = np.round(scores[top_n], 4)
    result.index = range(1, n + 1)

    if verbose:
        src = movie_profiles.iloc[idx]
        print(f"\nBecause you liked: '{movie_title}' ({int(src['year'])})")
        print(f"Genres: {src['genres_str']}")
        print()
        print(f"Top {n} recommendations:")
        print("-" * 65)
        print(f"{'#':<4} {'Title':<40} {'Year':<6} Similarity")
        print("-" * 65)
        for rank, row in result.iterrows():
            print(
                f"{rank:<4} {str(row['title'])[:39]:<40} "
                f"{str(int(row['year'])):<6} {row['similarity']:.4f}"
            )
        print("-" * 65)
    return result


def recommend_for_user(
    user_id,
    ratings: pd.DataFrame,
    movie_profiles: pd.DataFrame,
    embeddings: np.ndarray,
    n: int = 10,
    min_rating: float = 4.0,
    verbose: bool = True,
) -> pd.DataFrame | None:
    """Build a user profile (mean of liked-movie embeddings) and return top-N
    unseen recommendations."""
    liked_ids = ratings.loc[
        (ratings["userID"] == user_id) & (ratings["rating"] >= min_rating),
        "movieID",
    ].values

    liked_idx = movie_profiles.index[movie_profiles["movieID"].isin(liked_ids)].tolist()
    if not liked_idx:
        if verbose:
            print(f"User {user_id} has no liked movies (rating >= {min_rating}).")
        return None

    liked_titles = movie_profiles.loc[liked_idx, "title"].tolist()
    seen_ids = ratings.loc[ratings["userID"] == user_id, "movieID"].values

    profile = embeddings[liked_idx].mean(axis=0)
    profile = profile / np.maximum(np.linalg.norm(profile), 1e-9)

    scores = embeddings @ profile
    seen_mask = movie_profiles["movieID"].isin(seen_ids).values
    scores[seen_mask] = -1.0

    top_n = np.argsort(scores)[::-1][:n]
    result = movie_profiles[["title", "year", "genres_str"]].iloc[top_n].copy()
    result["score"] = np.round(scores[top_n], 4)
    result.index = range(1, n + 1)

    if verbose:
        print(f"\nUser {user_id} | rated {len(liked_titles)} movies >= {min_rating} stars")
        print(f"Sample liked: {liked_titles[:3]}")
        print()
        print(f"Top {n} recommendations:")
        print("-" * 65)
        for rank, row in result.iterrows():
            print(
                f"{rank:<4} {str(row['title'])[:39]:<40} "
                f"{str(int(row['year'])):<6} {row['score']:.4f}"
            )
        print("-" * 65)
    return result

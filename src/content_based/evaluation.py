"""Evaluation metrics for the content-based recommender."""

import numpy as np
import pandas as pd


def precision_at_k(
    user_id,
    ratings: pd.DataFrame,
    movie_profiles: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 10,
    min_rating: float = 4.0,
    test_frac: float = 0.2,
) -> float | None:
    """Hold out the last `test_frac` of a user's liked movies, recommend from
    the rest, and return Precision@k. Returns None if the user has fewer than
    5 liked movies."""
    liked = ratings.loc[
        (ratings["userID"] == user_id) & (ratings["rating"] >= min_rating),
        "movieID",
    ].tolist()
    if len(liked) < 5:
        return None

    n_test = max(1, int(len(liked) * test_frac))
    test_ids = set(liked[-n_test:])
    train_ids = liked[:-n_test]

    train_idx = movie_profiles.index[movie_profiles["movieID"].isin(train_ids)].tolist()
    if not train_idx:
        return None

    test_movie_ids = set(movie_profiles.loc[movie_profiles["movieID"].isin(test_ids), "movieID"].tolist())
    seen_ids = set(ratings.loc[ratings["userID"] == user_id, "movieID"].tolist())

    profile = embeddings[train_idx].mean(axis=0)
    profile = profile / np.maximum(np.linalg.norm(profile), 1e-9)

    scores = embeddings @ profile
    seen_mask = movie_profiles["movieID"].isin(seen_ids).values
    scores[seen_mask] = -1.0

    top_k_movie_ids = set(movie_profiles.iloc[np.argsort(scores)[::-1][:k]]["movieID"].tolist())
    return len(top_k_movie_ids & test_movie_ids) / k


def catalog_coverage(
    user_ids,
    ratings: pd.DataFrame,
    movie_profiles: pd.DataFrame,
    embeddings: np.ndarray,
    k: int = 10,
    min_rating: float = 4.0,
) -> float:
    """Fraction of catalog movies that appear in at least one user's top-k."""
    seen = set()
    for uid in user_ids:
        liked_ids = ratings.loc[
            (ratings["userID"] == uid) & (ratings["rating"] >= min_rating),
            "movieID",
        ].values
        liked_idx = movie_profiles.index[movie_profiles["movieID"].isin(liked_ids)].tolist()
        if not liked_idx:
            continue
        profile = embeddings[liked_idx].mean(axis=0)
        profile /= np.maximum(np.linalg.norm(profile), 1e-9)
        scores = embeddings @ profile
        top = np.argsort(scores)[::-1][:k]
        seen.update(movie_profiles.iloc[top]["movieID"].tolist())
    return len(seen) / len(movie_profiles)


def intra_list_diversity(rec_indices: list[int], embeddings: np.ndarray) -> float | None:
    """1 - mean pairwise cosine similarity among the recommended movies."""
    if len(rec_indices) < 2:
        return None
    sub = embeddings[rec_indices]
    sim = sub @ sub.T
    upper = sim[np.triu_indices_from(sim, k=1)]
    return float(1 - upper.mean())

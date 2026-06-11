"""
scripts/run_experiments.py
==========================
End-to-end MLflow experiment runner for the explainable-recsys project.

Run from the project root:
    python scripts/run_experiments.py

Then open the MLflow dashboard:
    mlflow ui  ->  http://localhost:5000
"""

from pathlib import Path

import mlflow
import pandas as pd

from src.models.collaborative_filtering import SGDMatrixFactorization, SVDCollaborativeFiltering
from src.models.popularity_recommender import PopularityRecommender

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR = Path("data")
EXPERIMENT_NAME = "explainable-recsys"
TEST_FRACTION = 0.2   # last 20% of ratings by time -> test set


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load pre-built feature parquet files from data/ and the raw movies
    table for title metadata.

    Returns
    -------
    movie_features       : one row per movieID, includes movie_avg_rating
                           and movie_rating_count (used by PopularityRecommender)
    interaction_features : one row per rating, includes userID, movieID,
                           interaction_rating, timestamp
    movies_metadata      : movieID + title (used by recommend() methods)
    """
    print("Loading movie features...")
    movie_features = pd.read_parquet(DATA_DIR / "movie_features.parquet")

    print("Loading interaction features...")
    interaction_features = pd.read_parquet(DATA_DIR / "interaction_features.parquet")

    print("Loading movies metadata (title)...")
    # The raw HetRec-2011 movies file — column 'id' maps to movieID.
    movies_raw = pd.read_csv(
        DATA_DIR / "movies.dat",
        sep="\t",
        encoding="latin-1",
        usecols=["id", "title"],
    ).rename(columns={"id": "movieID"})

    print(f"  movie_features:       {movie_features.shape}")
    print(f"  interaction_features: {interaction_features.shape}")
    print(f"  movies_metadata:      {movies_raw.shape}")

    return movie_features, interaction_features, movies_raw


# ---------------------------------------------------------------------------
# Train / test split
# ---------------------------------------------------------------------------

def temporal_train_test_split(
    interaction_features: pd.DataFrame,
    test_fraction: float = TEST_FRACTION,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Temporal split — sort interactions by timestamp, then take the last
    test_fraction as the test set.

    This avoids data leakage: the model is never trained on ratings that
    happened after the ratings it is asked to predict.
    """
    df = interaction_features.sort_values("timestamp").reset_index(drop=True)
    split_idx = int(len(df) * (1 - test_fraction))
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    return train_df, test_df


# ---------------------------------------------------------------------------
# Experiment runners — each creates exactly one MLflow run
# ---------------------------------------------------------------------------

def run_popularity_recommender(
    movie_features: pd.DataFrame,
    movies_metadata: pd.DataFrame,
) -> None:
    """Fit PopularityRecommender and print top-10 recommendations."""
    print("\n" + "=" * 50)
    print("PopularityRecommender")
    print("=" * 50)

    model = PopularityRecommender(min_votes=50, top_n=10)
    model.fit(movie_features, movies_metadata)

    print("\nTop 10 recommendations:")
    print(model.recommend().to_string(index=False))


def run_svd(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_factors: int = 50,
) -> None:
    """Train SVDCollaborativeFiltering, evaluate on test set."""
    print("\n" + "=" * 50)
    print(f"SVD  (n_factors={n_factors})")
    print("=" * 50)

    model = SVDCollaborativeFiltering(n_factors=n_factors, random_state=42)
    model.fit(train_df)
    results = model.evaluate(test_df)

    print(f"RMSE: {results['RMSE']:.4f}  |  MAE: {results['MAE']:.4f}")


def run_sgd(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    n_factors: int = 50,
    lr: float = 0.005,
    reg: float = 0.02,
    epochs: int = 20,
) -> None:
    """Train SGDMatrixFactorization, evaluate on test set."""
    print("\n" + "=" * 50)
    print(f"SGD  (n_factors={n_factors}, lr={lr}, reg={reg}, epochs={epochs})")
    print("=" * 50)

    model = SGDMatrixFactorization(
        n_factors=n_factors,
        learning_rate=lr,
        regularization=reg,
        epochs=epochs,
        random_state=42,
    )
    model.fit(train_df)
    results = model.evaluate(test_df)

    print(f"RMSE: {results['RMSE']:.4f}  |  MAE: {results['MAE']:.4f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # All runs will appear under this experiment in the MLflow UI
    mlflow.set_experiment(EXPERIMENT_NAME)

    # --- load data ----------------------------------------------------------
    movie_features, interaction_features, movies_metadata = load_data()

    # --- temporal train / test split ----------------------------------------
    train_df, test_df = temporal_train_test_split(interaction_features)
    print(f"\nTrain: {len(train_df):,} interactions | Test: {len(test_df):,} interactions")

    # --- run all three models -----------------------------------------------
    run_popularity_recommender(movie_features, movies_metadata)
    run_svd(train_df, test_df, n_factors=50)
    run_sgd(train_df, test_df, n_factors=50, lr=0.005, reg=0.02, epochs=20)

    print("\n✓ All experiments completed.")
    print("  Launch dashboard:  mlflow ui")
    print("  Then open:         http://localhost:5000")

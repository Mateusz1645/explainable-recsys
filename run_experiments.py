"""
MLflow Experiment Runner — explainable-recsys
=============================================
Run this script to train all models and track results in MLflow.

After running, launch the MLflow dashboard with:
    mlflow ui

Then open http://localhost:5000 in your browser.
"""

import mlflow

from collaborative_filtering import SGDMatrixFactorization, SVDCollaborativeFiltering
from popularity_recommender import PopularityRecommender

# ---------------------------------------------------------------------------
# Set the experiment name — all runs will be grouped under this name
# in the MLflow UI.
# ---------------------------------------------------------------------------
mlflow.set_experiment("explainable-recsys")


def run_popularity_recommender(movie_features_df, movies_metadata_df):
    """Train and log the PopularityRecommender."""
    print("\n=== PopularityRecommender ===")
    model = PopularityRecommender(min_votes=50, top_n=10)
    model.fit(movie_features_df, movies_metadata_df)
    print("Top 5 recommendations:")
    print(model.recommend(top_n=5))
    return model


def run_svd(train_df, test_df, movies_metadata_df, n_factors=50):
    """Train and evaluate SVDCollaborativeFiltering."""
    print(f"\n=== SVD (n_factors={n_factors}) ===")
    model = SVDCollaborativeFiltering(n_factors=n_factors, random_state=42)
    model.fit(train_df)
    results = model.evaluate(test_df)
    print(f"Test RMSE: {results['RMSE']:.4f} | MAE: {results['MAE']:.4f}")
    return model


def run_sgd(train_df, test_df, n_factors=50, lr=0.005, reg=0.02, epochs=20):
    """Train and evaluate SGDMatrixFactorization."""
    print(f"\n=== SGD (n_factors={n_factors}, lr={lr}, epochs={epochs}) ===")
    model = SGDMatrixFactorization(
        n_factors=n_factors,
        learning_rate=lr,
        regularization=reg,
        epochs=epochs,
        random_state=42,
    )
    model.fit(train_df)
    results = model.evaluate(test_df)
    print(f"Test RMSE: {results['RMSE']:.4f} | MAE: {results['MAE']:.4f}")
    return model


# ---------------------------------------------------------------------------
# MAIN — replace the placeholder data with your actual DataFrames
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # TODO: Load your data here, e.g.:
    #   from src.features.feature_engineering import build_all_features
    #   user_features, movie_features, interaction_features = build_all_features(data)
    #
    # For a quick test you can replace these with your actual DataFrames.

    # --- Example: run all three models ---
    # run_popularity_recommender(movie_features, movies_metadata)
    # run_svd(train_interactions, test_interactions, movies_metadata, n_factors=50)
    # run_sgd(train_interactions, test_interactions, n_factors=50, lr=0.005, epochs=20)

    print("Replace the placeholder data above and uncomment the function calls.")
    print("Then run:  mlflow ui   to see results at http://localhost:5000")

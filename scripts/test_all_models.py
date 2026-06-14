import argparse
import sys
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
SCRIPTS_DIR = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_experiments import temporal_train_test_split

from src.content_based.recommender import build_title_to_index, get_recommendations
from src.models.collaborative_filtering.collaborative_filtering import SGDMatrixFactorization, SVDCollaborativeFiltering
from src.models.popularity_recommender.popularity_recommender import PopularityRecommender


def load_data():
    movie_features = pd.read_parquet(DATA_DIR / "processed" / "movie_features.parquet")
    interactions = pd.read_parquet(DATA_DIR / "processed" / "interaction_features.parquet")
    movies = pd.read_csv(
        DATA_DIR / "raw" / "movies.dat",
        sep="\t",
        encoding="latin-1",
        usecols=["id", "title"],
    ).rename(columns={"id": "movieID"})
    return movie_features, interactions, movies


def run_popularity(movie_features, movies, top_n=10, min_votes=50):
    model = PopularityRecommender(min_votes=min_votes, top_n=top_n)
    model.fit(movie_features, movies)
    recs = model.recommend(top_n=top_n)
    if recs.empty:
        raise RuntimeError("Popularity model returned empty recommendations.")
    return {
        "status": "OK",
        "n_recommendations": int(len(recs)),
        "top1_weighted_rating": float(recs["weighted_rating"].iloc[0]),
    }


def run_svd(interactions, movies, top_n=10, n_factors=50):
    train_df, test_df = temporal_train_test_split(interactions, test_fraction=0.2)
    model = SVDCollaborativeFiltering(n_factors=n_factors, random_state=42)
    model.fit(train_df)
    metrics = model.evaluate(test_df)

    sample_user = int(train_df["userID"].iloc[0])
    recs = model.recommend(user_id=sample_user, movies_metadata_df=movies, top_n=top_n)
    if recs.empty:
        raise RuntimeError("SVD model returned empty recommendations.")

    return {
        "status": "OK",
        "rmse": float(metrics["RMSE"]),
        "mae": float(metrics["MAE"]),
        "n_recommendations": int(len(recs)),
    }


def run_sgd(interactions, movies, top_n=10, n_factors=50, lr=0.005, reg=0.02, epochs=20):
    train_df, test_df = temporal_train_test_split(interactions, test_fraction=0.2)
    model = SGDMatrixFactorization(
        n_factors=n_factors,
        learning_rate=lr,
        regularization=reg,
        epochs=epochs,
        random_state=42,
    )
    model.fit(train_df)
    metrics = model.evaluate(test_df)

    sample_user = int(train_df["userID"].iloc[0])
    recs = model.recommend(user_id=sample_user, movies_metadata_df=movies, top_n=top_n)
    if recs.empty:
        raise RuntimeError("SGD model returned empty recommendations.")

    return {
        "status": "OK",
        "rmse": float(metrics["RMSE"]),
        "mae": float(metrics["MAE"]),
        "n_recommendations": int(len(recs)),
    }


def run_content(top_n=10):
    profiles_path = DATA_DIR / "processed" / "movie_content_profiles.csv"
    emb_path = DATA_DIR / "processed" / "movie_embeddings.npy"
    if not profiles_path.exists() or not emb_path.exists():
        return {
            "status": "SKIPPED",
            "reason": "Missing movie_content_profiles.csv or movie_embeddings.npy",
        }

    profiles = pd.read_csv(profiles_path)
    emb = np.load(emb_path)
    title_to_idx = build_title_to_index(profiles)

    if len(title_to_idx) == 0:
        raise RuntimeError("No titles found in content profiles.")

    sample_title = str(title_to_idx.index[0])
    recs = get_recommendations(
        movie_title=sample_title,
        movie_profiles=profiles,
        embeddings=emb,
        title_to_index=title_to_idx,
        n=top_n,
        verbose=False,
    )
    if recs is None or recs.empty:
        raise RuntimeError("Content-based model returned no recommendations.")

    return {
        "status": "OK",
        "sample_title": sample_title,
        "n_recommendations": int(len(recs)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=["all", "popularity", "svd", "sgd", "content"], default="all")
    parser.add_argument("--top-n", type=int, default=10)
    args = parser.parse_args()

    movie_features, interactions, movies = load_data()
    selected = ["popularity", "svd", "sgd", "content"] if args.model == "all" else [args.model]

    results = {}
    failed = False

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("test_all_models")

    if mlflow.active_run() is not None:
        mlflow.end_run()

    for name in selected:
        try:
            if mlflow.active_run() is not None:
                mlflow.end_run()

            if name == "popularity":
                res = run_popularity(movie_features, movies, top_n=args.top_n)
            elif name == "svd":
                res = run_svd(interactions, movies, top_n=args.top_n)
            elif name == "sgd":
                res = run_sgd(interactions, movies, top_n=args.top_n)
            else:
                res = run_content(top_n=args.top_n)

            results[name] = res
            if res.get("status") not in ("OK", "SKIPPED"):
                failed = True

        except Exception as e:
            results[name] = {"status": "FAILED", "error": str(e)}
            failed = True

        finally:
            if mlflow.active_run() is not None:
                mlflow.end_run()

    print(pd.DataFrame(results).T.to_string())

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
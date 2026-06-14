import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.content_based.recommender import build_title_to_index, get_recommendations
from src.models.collaborative_filtering.collaborative_filtering import SGDMatrixFactorization, SVDCollaborativeFiltering
from src.models.popularity_recommender.popularity_recommender import PopularityRecommender

LIGHT_MODE = os.getenv("LIGHT_MODE", "true").lower() == "true"

st.set_page_config(page_title="Explainable Recsys Demo", page_icon="🎬", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"


@st.cache_data
def load_movies_and_features():
    movie_features = pd.read_parquet(DATA_DIR / "processed" / "movie_features.parquet")
    movies = pd.read_csv(
        DATA_DIR / "raw" / "movies.dat",
        sep="\t",
        encoding="latin-1",
        usecols=["id", "title"],
    ).rename(columns={"id": "movieID"})
    return movie_features, movies


@st.cache_data
def load_interactions(max_rows: int = 250_000):
    cols = ["userID", "movieID", "interaction_rating", "timestamp"]
    df = pd.read_parquet(
        DATA_DIR / "processed" / "interaction_features.parquet",
        columns=cols,
    )

    df["userID"] = pd.to_numeric(df["userID"], downcast="integer")
    df["movieID"] = pd.to_numeric(df["movieID"], downcast="integer")
    df["interaction_rating"] = pd.to_numeric(df["interaction_rating"], downcast="float")
    df["timestamp"] = pd.to_numeric(df["timestamp"], downcast="integer")

    if len(df) > max_rows:
        df = df.sort_values("timestamp").tail(max_rows).reset_index(drop=True)

    return df


def temporal_split(df: pd.DataFrame, test_fraction: float = 0.2):
    d = df.sort_values("timestamp").reset_index(drop=True)
    cut = int(len(d) * (1 - test_fraction))
    return d.iloc[:cut].copy(), d.iloc[cut:].copy()


@st.cache_resource
def fit_popularity(min_votes: int, top_n: int):
    movie_features, movies = load_movies_and_features()
    model = PopularityRecommender(min_votes=min_votes, top_n=top_n)
    model.fit(movie_features, movies)
    return model


@st.cache_resource
def fit_svd(n_factors: int):
    interactions = load_interactions(max_rows=200_000)
    _, movies = load_movies_and_features()
    train_df, test_df = temporal_split(interactions)
    model = SVDCollaborativeFiltering(n_factors=n_factors, random_state=42)
    model.fit(train_df)
    metrics = model.evaluate(test_df)
    return model, train_df, movies, metrics


@st.cache_resource
def fit_sgd(n_factors: int, lr: float, reg: float, epochs: int):
    interactions = load_interactions(max_rows=200_000)
    _, movies = load_movies_and_features()
    train_df, test_df = temporal_split(interactions)
    model = SGDMatrixFactorization(
        n_factors=n_factors,
        learning_rate=lr,
        regularization=reg,
        epochs=epochs,
        random_state=42,
    )
    model.fit(train_df)
    metrics = model.evaluate(test_df)
    return model, train_df, movies, metrics


@st.cache_data
def load_content_assets():
    profiles_path = DATA_DIR / "processed" / "movie_content_profiles.csv"
    emb_path = DATA_DIR / "processed" / "movie_embeddings.npy"
    if not profiles_path.exists() or not emb_path.exists():
        return None, None, None
    profiles = pd.read_csv(profiles_path)
    emb = np.load(emb_path)
    title_to_idx = build_title_to_index(profiles)
    return profiles, emb, title_to_idx


st.title("🎬 Explainable Recsys — Multi-Model Demo")

model_options = ["Popularity", "Content-Based (Embeddings)"]
if not LIGHT_MODE:
    model_options = ["Popularity", "SVD Collaborative Filtering", "SGD Matrix Factorization", "Content-Based (Embeddings)"]

model_type = st.selectbox("Select model", model_options)
top_n = st.slider("Top N", 5, 20, 10, 1)

if model_type == "Popularity":
    min_votes = st.slider("min_votes", 10, 300, 50, 10)
    model = fit_popularity(min_votes=min_votes, top_n=top_n)
    recs = model.recommend(top_n=top_n)
    recs["weighted_rating"] = recs["weighted_rating"].round(4)
    recs["movie_avg_rating"] = recs["movie_avg_rating"].round(3)

    st.subheader("Recommendations")
    st.dataframe(
        recs[["title", "movie_avg_rating", "movie_rating_count", "weighted_rating"]],
        use_container_width=True,
        hide_index=True,
    )
    st.info("Explanation: ranking based on IMDb-style weighted rating (R, v, m, C).")

elif model_type == "SVD Collaborative Filtering":
    if LIGHT_MODE:
        st.error("This model is disabled in cloud LIGHT_MODE due to memory limits.")
        st.stop()

    n_factors = st.slider("n_factors", 20, 150, 50, 10)
    model, train_df, movies_meta, metrics = fit_svd(n_factors=n_factors)

    user_ids = sorted(train_df["userID"].unique().tolist())
    user_id = st.selectbox("Select userID", user_ids, index=0)
    recs = model.recommend(user_id=user_id, movies_metadata_df=movies_meta, top_n=top_n)

    st.subheader("Recommendations")
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
    st.metric("MAE", f"{metrics['MAE']:.4f}")
    st.info("Explanation: recommendations are based on latent factors (user-item matrix decomposition).")

elif model_type == "SGD Matrix Factorization":
    if LIGHT_MODE:
        st.error("This model is disabled in cloud LIGHT_MODE due to memory limits.")
        st.stop()

    n_factors = st.slider("n_factors", 20, 150, 50, 10)
    lr = st.slider("learning_rate", 0.001, 0.02, 0.005, 0.001)
    reg = st.slider("regularization", 0.001, 0.1, 0.02, 0.001)
    epochs = st.slider("epochs", 5, 40, 20, 1)

    model, train_df, movies_meta, metrics = fit_sgd(n_factors=n_factors, lr=lr, reg=reg, epochs=epochs)

    user_ids = sorted(train_df["userID"].unique().tolist())
    user_id = st.selectbox("Select userID", user_ids, index=0)
    recs = model.recommend(user_id=user_id, movies_metadata_df=movies_meta, top_n=top_n)

    st.subheader("Recommendations")
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.metric("RMSE", f"{metrics['RMSE']:.4f}")
    st.metric("MAE", f"{metrics['MAE']:.4f}")
    st.info("Explanation: predicted rating = global mean + user/movie bias + latent vector dot product.")

else:
    profiles, emb, title_to_idx = load_content_assets()
    if profiles is None:
        st.error("Missing content-based files: data/processed/movie_content_profiles.csv and/or movie_embeddings.npy")
    else:
        titles = sorted(title_to_idx.index.tolist())
        selected_title = st.selectbox("Select movie", titles, index=0)
        recs = get_recommendations(
            movie_title=selected_title,
            movie_profiles=profiles,
            embeddings=emb,
            title_to_index=title_to_idx,
            n=top_n,
            verbose=False,
        )
        if recs is not None:
            st.subheader("Recommendations")
            st.dataframe(recs.reset_index(drop=True), use_container_width=True, hide_index=True)
            st.info("Explanation: cosine similarity in embedding space.")

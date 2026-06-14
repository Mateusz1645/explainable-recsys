import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from src.content_based.recommender import build_title_to_index, get_recommendations
from src.models.popularity_recommender.popularity_recommender import PopularityRecommender

LIGHT_MODE = os.getenv("LIGHT_MODE", "true").lower() == "true"

st.set_page_config(page_title="Explainable Recsys Demo", page_icon="🎬", layout="wide")

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"


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


@st.cache_resource
def fit_popularity(min_votes: int, top_n: int):
    movie_features, movies = load_movies_and_features()
    model = PopularityRecommender(min_votes=min_votes, top_n=top_n)
    model.fit(movie_features, movies)
    return model


@st.cache_resource
def load_svd_model():
    p = ARTIFACTS_DIR / "svd_model.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_sgd_model():
    p = ARTIFACTS_DIR / "sgd_model.pkl"
    if not p.exists():
        return None
    with open(p, "rb") as f:
        return pickle.load(f)


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

profiles_path = DATA_DIR / "processed" / "movie_content_profiles.csv"
emb_path = DATA_DIR / "processed" / "movie_embeddings.npy"
content_available = profiles_path.exists() and emb_path.exists()

model_options = ["Popularity"]
if not LIGHT_MODE:
    model_options += ["SVD Collaborative Filtering", "SGD Matrix Factorization"]
if content_available:
    model_options += ["Content-Based (Embeddings)"]

model_type = st.selectbox("Select model", model_options)
top_n = st.slider("Top N", 5, 20, 10, 1)

if not content_available:
    st.caption("Content-Based model is hidden because embedding files are missing.")

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

    model = load_svd_model()
    if model is None:
        st.error("Missing artifacts/svd_model.pkl. Train locally with: python scripts/train_and_save_models.py")
        st.stop()

    _, movies_meta = load_movies_and_features()
    user_ids = sorted(model.train_df["userID"].unique().tolist())
    user_id = st.selectbox("Select userID", user_ids, index=0)
    recs = model.recommend(user_id=user_id, movies_metadata_df=movies_meta, top_n=top_n)

    st.subheader("Recommendations")
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.info("Model loaded from pre-trained artifact (no runtime training).")

elif model_type == "SGD Matrix Factorization":
    if LIGHT_MODE:
        st.error("This model is disabled in cloud LIGHT_MODE due to memory limits.")
        st.stop()

    model = load_sgd_model()
    if model is None:
        st.error("Missing artifacts/sgd_model.pkl. Train locally with: python scripts/train_and_save_models.py")
        st.stop()

    _, movies_meta = load_movies_and_features()
    user_ids = sorted(model.train_df["userID"].unique().tolist())
    user_id = st.selectbox("Select userID", user_ids, index=0)
    recs = model.recommend(user_id=user_id, movies_metadata_df=movies_meta, top_n=top_n)

    st.subheader("Recommendations")
    st.dataframe(recs, use_container_width=True, hide_index=True)
    st.info("Model loaded from pre-trained artifact (no runtime training).")

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
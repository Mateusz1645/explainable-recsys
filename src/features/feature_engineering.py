"""
Feature Engineering for Explainable Movie Recommendation System
===============================================================
Dataset : HetRec-2011 MovieLens
Tables  : movies, movie_actors, movie_countries, movie_directors,
          movie_genres, movie_locations, movie_tags, tags,
          user_ratedmovies, user_taggedmovies
Author  : (your name)

Usage
-----
Assumes the data dictionary produced by load_data_tables() is available:

    data = load_data_tables(file_map)
    from feature_engineering import build_all_features
    user_features, movie_features, interaction_features = build_all_features(data)

SHAP / LIME note
----------------
Every feature is named in plain English (e.g. ``user_avg_rating``,
``movie_genre_count``) so that SHAP waterfall / summary plots are
human-readable without any post-processing.

Leakage guard
-------------
All aggregations over user_ratedmovies must later be recomputed using
ONLY the training split.  The functions here are written split-agnostic
so you can call them on df_train once you have a temporal split.
"""

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Last rating timestamp in the dataset  – used to compute recency features.
DATASET_END_DATE = pd.Timestamp("2009-01-05")

# Genres that appear in movie_genres (complete list from EDA).
ALL_GENRES = [
    "Action", "Adventure", "Animation", "Children", "Comedy",
    "Crime", "Documentary", "Drama", "Fantasy", "Film-Noir",
    "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
    "Thriller", "War", "Western",
]

# Country grouping: countries with fewer than this many movies → "Other".
RARE_COUNTRY_THRESHOLD = 50

# Director grouping: directors with fewer than this many movies → "Other".
RARE_DIRECTOR_THRESHOLD = 3

# Cap cast size at this percentile to handle the 220-actor outlier.
CAST_SIZE_CAP = 50


# ===========================================================================
# HELPER: build a proper timestamp column from HetRec date parts
# ===========================================================================

def _build_timestamp(df: pd.DataFrame) -> pd.Series:
    """
    HetRec-2011 stores dates as 6 separate integer columns:
    date_year, date_month, date_day, date_hour, date_minute, date_second.

    Returns a single pd.Timestamp Series named 'timestamp'.
    """
    return pd.to_datetime(
        {
            "year":   df["date_year"],
            "month":  df["date_month"],
            "day":    df["date_day"],
            "hour":   df["date_hour"],
            "minute": df["date_minute"],
            "second": df["date_second"],
        }
    )


# ===========================================================================
# GROUP 1 — USER FEATURES
# ===========================================================================
# Source tables: user_ratedmovies, user_taggedmovies
#
# Why this matters for SHAP: user-level features explain *who* the user is.
# When SHAP decomposes a prediction, features like user_avg_rating reveal
# whether the model is reacting to a lenient vs. strict rater, making the
# explanation immediately interpretable to a non-technical audience.
# ===========================================================================


def build_user_rating_features(
    user_ratedmovies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate explicit rating behaviour per user.

    Features produced
    -----------------
    user_rating_count        : total number of ratings  → activity level
    user_avg_rating          : mean rating              → leniency bias
    user_rating_std          : std dev of ratings       → taste consistency
    user_min_rating          : lowest rating ever given → harshness floor
    user_max_rating          : highest rating given     → generosity ceiling
    user_rating_range        : max − min                → scale usage breadth
    user_pct_high_ratings    : share of ratings ≥ 4.0   → positivity tendency
    user_first_rating_date   : earliest rating date     → tenure start
    user_last_rating_date    : most recent rating date  → recency
    user_activity_span_days  : last − first (days)      → longevity
    user_days_since_last_rating : DATASET_END_DATE − last (days) → recency
    user_avg_rating_log_count: log1p(count) × avg_rating → engagement × quality
    """
    df = user_ratedmovies_df.copy()
    df["timestamp"] = _build_timestamp(df)

    # --- core rating statistics -------------------------------------------
    agg = df.groupby("userID").agg(
        user_rating_count=("rating", "count"),
        user_avg_rating=("rating", "mean"),
        user_rating_std=("rating", "std"),
        user_min_rating=("rating", "min"),
        user_max_rating=("rating", "max"),
        user_first_rating_date=("timestamp", "min"),
        user_last_rating_date=("timestamp", "max"),
    ).reset_index()

    # std is NaN for users with a single rating – fill with 0
    agg["user_rating_std"] = agg["user_rating_std"].fillna(0.0)

    # --- derived features -------------------------------------------------
    agg["user_rating_range"] = (
        agg["user_max_rating"] - agg["user_min_rating"]
    )

    high_rating_counts = (
        df[df["rating"] >= 4.0]
        .groupby("userID")
        .size()
        .rename("_high_count")
    )
    agg = agg.merge(high_rating_counts, on="userID", how="left")
    agg["_high_count"] = agg["_high_count"].fillna(0)
    agg["user_pct_high_ratings"] = (
        agg["_high_count"] / agg["user_rating_count"]
    )
    agg = agg.drop(columns=["_high_count"])

    # --- temporal features ------------------------------------------------
    agg["user_activity_span_days"] = (
        agg["user_last_rating_date"] - agg["user_first_rating_date"]
    ).dt.days.fillna(0)

    agg["user_days_since_last_rating"] = (
        DATASET_END_DATE - agg["user_last_rating_date"]
    ).dt.days

    # --- composite feature ------------------------------------------------
    # Useful for tree models and SHAP: captures "how active AND how positive".
    agg["user_avg_rating_log_count"] = (
        agg["user_avg_rating"] * np.log1p(agg["user_rating_count"])
    )

    # Drop raw date columns – models should not see raw Timestamps.
    agg = agg.drop(
        columns=["user_first_rating_date", "user_last_rating_date"]
    )

    return agg


def build_user_tag_features(
    user_taggedmovies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate implicit tagging behaviour per user.

    Features produced
    -----------------
    user_tag_event_count      : total tagging events   → implicit engagement
    user_unique_movies_tagged : unique movies tagged    → taste breadth
    user_unique_tags_used     : unique tags applied     → vocabulary breadth
    user_tagging_intensity    : events / unique movies  → depth of engagement

    WHY: Tagging is an implicit feedback signal — a user who tags movies
    is more engaged.  Even if your final model uses only ratings, tagging
    features improve cold-start and SHAP explanations ("this user is a
    heavy tagger").
    """
    agg = user_taggedmovies_df.groupby("userID").agg(
        user_tag_event_count=("tagID", "count"),
        user_unique_movies_tagged=("movieID", "nunique"),
        user_unique_tags_used=("tagID", "nunique"),
    ).reset_index()

    agg["user_tagging_intensity"] = (
        agg["user_tag_event_count"] / agg["user_unique_movies_tagged"]
    )

    return agg


def build_user_features(
    user_ratedmovies_df: pd.DataFrame,
    user_taggedmovies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all user feature blocks into a single DataFrame indexed by userID.

    Users without any tagging history get 0 for tagging features.
    """
    rating_feats = build_user_rating_features(user_ratedmovies_df)
    tag_feats = build_user_tag_features(user_taggedmovies_df)

    user_features = rating_feats.merge(tag_feats, on="userID", how="left")

    # Users who never tagged anything → fill tag features with 0
    tag_cols = [
        "user_tag_event_count",
        "user_unique_movies_tagged",
        "user_unique_tags_used",
        "user_tagging_intensity",
    ]
    user_features[tag_cols] = user_features[tag_cols].fillna(0)

    return user_features


# ===========================================================================
# GROUP 2 — ITEM (MOVIE) FEATURES
# ===========================================================================
# Source tables: movies, movie_actors, movie_genres, movie_directors,
#                movie_countries, movie_tags, movie_locations,
#                user_ratedmovies (aggregated)
#
# Why this matters for SHAP: item features explain *what* is being
# recommended.  A SHAP plot showing ``genre_Drama=1`` or
# ``movie_rt_audience_rating=8.2`` tells users exactly which movie
# properties drove the recommendation — crucial for explainability.
# ===========================================================================


def build_movie_popularity_features(
    user_ratedmovies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate per-movie statistics from the interaction log.

    Features produced
    -----------------
    movie_rating_count      : total ratings received     → popularity
    movie_log_rating_count  : log1p(count)               → compressed popularity
    movie_avg_rating        : mean rating                → perceived quality
    movie_rating_std        : std of ratings             → polarisation
    movie_pct_high_ratings  : share of ratings ≥ 4.0    → crowd approval

    WHY: Even content-based models benefit from popularity priors.
    log1p compression is applied because your EDA confirmed a heavy tail
    (p99 movie rating count ≈ 914 vs median ≈ low tens).
    """
    df = user_ratedmovies_df

    agg = df.groupby("movieID").agg(
        movie_rating_count=("rating", "count"),
        movie_avg_rating=("rating", "mean"),
        movie_rating_std=("rating", "std"),
    ).reset_index()

    agg["movie_rating_std"] = agg["movie_rating_std"].fillna(0.0)
    agg["movie_log_rating_count"] = np.log1p(agg["movie_rating_count"])

    high_counts = (
        df[df["rating"] >= 4.0]
        .groupby("movieID")
        .size()
        .rename("_high")
    )
    agg = agg.merge(high_counts, on="movieID", how="left")
    agg["_high"] = agg["_high"].fillna(0)
    agg["movie_pct_high_ratings"] = agg["_high"] / agg["movie_rating_count"]
    agg = agg.drop(columns=["_high"])

    return agg


def build_movie_metadata_features(movies_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract features from the core movies table (21 columns).

    Features produced
    -----------------
    movie_year              : release year          → era/recency signal
    movie_age               : DATASET_END_DATE.year − year → movie age
    movie_has_rt_data       : 1 if rtID is not null → data completeness flag
    movie_rt_critics_rating : rtAllCriticsRating    → critic sentiment
    movie_rt_audience_rating: rtAudienceRating      → audience sentiment
    movie_rt_critics_score  : rtAllCriticsScore (%) → freshness
    movie_rt_audience_score : rtAudienceScore (%)   → audience approval

    WHY: Your EDA notes that rtID has significant missingness.
    The binary ``movie_has_rt_data`` flag doubles as a quality signal AND
    makes the missingness pattern visible to SHAP/LIME.
    """
    df = movies_df[
        [
            "id", "year",
            "rtID",
            "rtAllCriticsRating", "rtAllCriticsScore",
            "rtAudienceRating",   "rtAudienceScore",
        ]
    ].copy().rename(columns={"id": "movieID"})

    df["movie_year"] = df["year"].fillna(0).astype(int)
    df["movie_age"] = DATASET_END_DATE.year - df["movie_year"]
    df["movie_age"] = df["movie_age"].clip(lower=0)

    df["movie_has_rt_data"] = df["rtID"].notna().astype(int)

    df = df.rename(
        columns={
            "rtAllCriticsRating": "movie_rt_critics_rating",
            "rtAllCriticsScore":  "movie_rt_critics_score",
            "rtAudienceRating":   "movie_rt_audience_rating",
            "rtAudienceScore":    "movie_rt_audience_score",
        }
    )

    # For rows with no RT data, fill numerical scores with dataset median.
    # This is safer than 0 (which would look like a terrible score to the model).
    for col in [
        "movie_rt_critics_rating",
        "movie_rt_critics_score",
        "movie_rt_audience_rating",
        "movie_rt_audience_score",
    ]:
        median_val = df.loc[df["movie_has_rt_data"] == 1, col].median()
        df[col] = df[col].fillna(median_val)

    return df[
        [
            "movieID", "movie_year", "movie_age", "movie_has_rt_data",
            "movie_rt_critics_rating", "movie_rt_critics_score",
            "movie_rt_audience_rating", "movie_rt_audience_score",
        ]
    ]


def build_movie_genre_features(movie_genres_df: pd.DataFrame) -> pd.DataFrame:
    """
    Produce multi-hot genre encodings + a genre count feature.

    Features produced
    -----------------
    movie_genre_count     : number of genres assigned  → content richness
    genre_<GenreName>     : 1/0 multi-hot per genre    → content signal

    WHY: Your EDA confirms multi-label genre assignment (avg ~2 per movie),
    so multi-hot is the correct encoding — NOT label encoding.
    Multi-hot columns are SHAP-friendly: "genre_Drama=1 contributed +0.3
    to this recommendation" is immediately interpretable.
    """
    # Count genres per movie
    genre_counts = (
        movie_genres_df
        .groupby("movieID")
        .size()
        .reset_index(name="movie_genre_count")
    )

    # Multi-hot: pivot genre values into binary columns
    movie_genres_df = movie_genres_df.copy()
    movie_genres_df["_val"] = 1
    genre_pivot = (
        movie_genres_df
        .pivot_table(
            index="movieID",
            columns="genre",
            values="_val",
            aggfunc="max",
            fill_value=0,
        )
        .reset_index()
    )

    # Rename columns to be model-friendly and SHAP-readable
    genre_pivot.columns.name = None
    genre_pivot = genre_pivot.rename(
        columns={g: f"genre_{g}" for g in ALL_GENRES if g in genre_pivot.columns}
    )

    # Ensure ALL expected genre columns exist (some may be absent)
    for genre in ALL_GENRES:
        col = f"genre_{genre}"
        if col not in genre_pivot.columns:
            genre_pivot[col] = 0

    result = genre_counts.merge(genre_pivot, on="movieID", how="left")
    return result


def build_movie_actor_features(
    movie_actors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Aggregate cast-level features per movie.

    Features produced
    -----------------
    movie_cast_size          : total actors listed      → production scale
    movie_cast_size_capped   : min(cast_size, CAP)      → outlier-robust version
    movie_top_actor_rank_min : min ranking value        → lead actor prominence
    movie_top_actor_rank_mean: mean ranking             → ensemble vs. lead
    movie_has_rank_conflict  : 1 if any rank duplication → data quality flag

    WHY: Your EDA found a median cast of 17 but a tail up to 220.
    Capping at the 90th percentile (50) prevents outliers from dominating
    tree splits.  The rank_conflict flag keeps the quality signal visible
    to SHAP rather than silently masking it.
    """
    df = movie_actors_df.copy()

    # Flag rank conflicts per movie (same rank assigned to 2+ actors)
    rank_dup_flags = (
        df.groupby("movieID")["ranking"]
        .apply(lambda x: int(x.duplicated().any()))
        .reset_index(name="movie_has_rank_conflict")
    )

    agg = df.groupby("movieID").agg(
        movie_cast_size=("actorID", "count"),
        movie_top_actor_rank_min=("ranking", "min"),
        movie_top_actor_rank_mean=("ranking", "mean"),
    ).reset_index()

    agg["movie_cast_size_capped"] = agg["movie_cast_size"].clip(
        upper=CAST_SIZE_CAP
    )
    agg = agg.merge(rank_dup_flags, on="movieID", how="left")
    agg["movie_has_rank_conflict"] = (
        agg["movie_has_rank_conflict"].fillna(0).astype(int)
    )

    return agg


def build_movie_director_features(
    movie_directors_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Produce director-level features per movie.

    Features produced
    -----------------
    movie_has_director       : 1 if director is known   → data completeness
    movie_director_experience: # movies this director made → career depth

    WHY: Director experience is a strong content signal — prolific
    directors tend to produce more consistently rated movies.  The binary
    flag preserves the 42-missing-movie signal your EDA found.
    """
    # Director experience = how many movies each director appears in
    dir_exp = (
        movie_directors_df
        .groupby("directorID")
        .size()
        .reset_index(name="director_experience")
    )

    df = movie_directors_df.merge(dir_exp, on="directorID", how="left")

    # If a movie has multiple directors, take the max experience (primary director proxy)
    agg = df.groupby("movieID").agg(
        movie_director_experience=("director_experience", "max"),
    ).reset_index()
    agg["movie_has_director"] = 1

    return agg


def build_movie_country_features(
    movie_countries_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Encode country as a frequency-based categorical.

    Features produced
    -----------------
    movie_country_encoded: label-encoded country (rare countries → 'Other')

    WHY: Your EDA confirmed exactly 1 country per movie (not multi-label).
    Rare countries are grouped into 'Other' to prevent high cardinality
    from harming tree models.  A string column is returned — apply your
    pipeline's OrdinalEncoder or TargetEncoder downstream.
    """
    df = movie_countries_df.copy()
    df["country"] = df["country"].fillna("Unknown")

    country_counts = df["country"].value_counts()
    rare_countries = country_counts[
        country_counts < RARE_COUNTRY_THRESHOLD
    ].index

    df["movie_country_encoded"] = df["country"].where(
        ~df["country"].isin(rare_countries), other="Other"
    )

    return df[["movieID", "movie_country_encoded"]]


def build_movie_tag_features(movie_tags_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate semantic tagging signals per movie.

    Features produced
    -----------------
    movie_unique_tag_count   : distinct tags applied       → semantic richness
    movie_tag_weight_sum     : sum of tagWeights (total assignments) → tag volume
    movie_tag_weight_mean    : mean tagWeight per tag       → tag concentration
    movie_log_tag_weight_sum : log1p(weight_sum)           → compressed volume

    WHY: tagWeight is a raw assignment count (as confirmed in your EDA).
    High tag weight sum signals a well-described, popular movie.
    log1p compression handles the heavy tail your EDA confirmed.
    """
    agg = movie_tags_df.groupby("movieID").agg(
        movie_unique_tag_count=("tagID", "count"),
        movie_tag_weight_sum=("tagWeight", "sum"),
        movie_tag_weight_mean=("tagWeight", "mean"),
    ).reset_index()

    agg["movie_log_tag_weight_sum"] = np.log1p(agg["movie_tag_weight_sum"])

    return agg


def build_movie_location_features(
    movie_locations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Derive a location depth score per movie.

    Features produced
    -----------------
    movie_location_depth: number of non-null location levels (0-4)
                          → metadata richness proxy

    WHY: Your EDA confirmed missingness increases with depth (location4
    is sparsest).  Depth is a simple, SHAP-interpretable proxy for how
    well-documented a movie's filming locations are.
    """
    df = movie_locations_df.copy()
    loc_cols = ["location1", "location2", "location3", "location4"]

    df["movie_location_depth"] = df[loc_cols].notna().sum(axis=1)

    agg = df.groupby("movieID")["movie_location_depth"].max().reset_index()

    return agg


def build_movie_features(
    movies_df: pd.DataFrame,
    movie_actors_df: pd.DataFrame,
    movie_genres_df: pd.DataFrame,
    movie_directors_df: pd.DataFrame,
    movie_countries_df: pd.DataFrame,
    movie_tags_df: pd.DataFrame,
    movie_locations_df: pd.DataFrame,
    user_ratedmovies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge all movie feature blocks into a single DataFrame indexed by movieID.

    Movies with no director, no tags, or no locations get sensible fill values.
    """
    base = movies_df[["id"]].rename(columns={"id": "movieID"})

    feature_blocks = [
        build_movie_metadata_features(movies_df),
        build_movie_popularity_features(user_ratedmovies_df),
        build_movie_genre_features(movie_genres_df),
        build_movie_actor_features(movie_actors_df),
        build_movie_director_features(movie_directors_df),
        build_movie_country_features(movie_countries_df),
        build_movie_tag_features(movie_tags_df),
        build_movie_location_features(movie_locations_df),
    ]

    movie_features = base
    for block in feature_blocks:
        movie_features = movie_features.merge(block, on="movieID", how="left")

    # --- fill missing values for optional metadata -----------------------
    movie_features["movie_has_director"] = (
        movie_features["movie_has_director"].fillna(0).astype(int)
    )
    movie_features["movie_director_experience"] = (
        movie_features["movie_director_experience"].fillna(0)
    )

    for col in ["movie_unique_tag_count", "movie_tag_weight_sum",
                "movie_tag_weight_mean", "movie_log_tag_weight_sum"]:
        movie_features[col] = movie_features[col].fillna(0)

    movie_features["movie_location_depth"] = (
        movie_features["movie_location_depth"].fillna(0).astype(int)
    )
    movie_features["movie_country_encoded"] = (
        movie_features["movie_country_encoded"].fillna("Unknown")
    )

    return movie_features


# ===========================================================================
# GROUP 3 — USER-ITEM INTERACTION FEATURES
# ===========================================================================
# Source tables: user_ratedmovies + precomputed user_features + movie_features
#
# Why this matters for SHAP: interaction features capture the *relationship*
# between user and item at the moment of the interaction.  Features like
# ``interaction_rating_vs_user_avg`` tell SHAP "the user rated this movie
# higher than they normally rate" — this kind of relative signal is the most
# powerful for explaining *why* a specific recommendation was made.
# ===========================================================================


def build_interaction_features(
    user_ratedmovies_df: pd.DataFrame,
    user_features: pd.DataFrame,
    movie_features: pd.DataFrame,
    movies_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create user-item interaction features by joining the rating log with
    precomputed user and movie feature tables.

    Features produced
    -----------------
    interaction_rating            : raw rating (target or feature depending on model)
    interaction_movie_age_at_rating: movie age in years at the time of the rating
    interaction_rating_vs_user_avg : rating − user_avg_rating  → personal deviation
    interaction_rating_vs_movie_avg: rating − movie_avg_rating → crowd deviation
    interaction_user_rating_sequence: position in user's rating history (1 = first)
    interaction_is_early_rater    : 1 if user rated movie within 1 yr of release
    interaction_genre_affinity_*  : user's avg rating per genre (joined from profile)

    WHY: These features capture the *context* of an interaction, not just
    static user/item properties.  They are essential for LIME which
    perturbs individual interaction records to explain single predictions.
    """
    df = user_ratedmovies_df.copy()
    df["timestamp"] = _build_timestamp(df)

    # --- 1. merge user and movie summary stats ----------------------------
    user_cols = ["userID", "user_avg_rating", "user_rating_count"]
    movie_cols = [
        "movieID", "movie_avg_rating", "movie_year",
    ]
    df = df.merge(user_features[user_cols], on="userID", how="left")
    df = df.merge(movie_features[movie_cols], on="movieID", how="left")

    # --- 2. rating deviations ---------------------------------------------
    df["interaction_rating"] = df["rating"]
    df["interaction_rating_vs_user_avg"] = (
        df["rating"] - df["user_avg_rating"]
    )
    df["interaction_rating_vs_movie_avg"] = (
        df["rating"] - df["movie_avg_rating"]
    )

    # --- 3. movie age at the time of the rating ---------------------------
    df["interaction_movie_age_at_rating"] = (
        df["timestamp"].dt.year - df["movie_year"]
    ).clip(lower=0)

    # --- 4. is_early_rater -----------------------------------------------
    # 1 if the user rated the movie within 1 year of its release
    df["interaction_is_early_rater"] = (
        df["interaction_movie_age_at_rating"] <= 1
    ).astype(int)

    # --- 5. user rating sequence ------------------------------------------
    # Position of this rating in the user's personal history (chronological).
    # Useful for sequential models and detecting "rating fatigue".
    df = df.sort_values(["userID", "timestamp"])
    df["interaction_user_rating_sequence"] = (
        df.groupby("userID").cumcount() + 1
    )

    # --- 6. genre affinity features per interaction -----------------------
    # For each interaction, look up the user's average rating for each
    # genre of the movie being rated.  This creates a personalised
    # content-based signal at interaction time.
    genre_affinity = _build_genre_affinity(user_ratedmovies_df, movie_features)
    df = df.merge(genre_affinity, on="userID", how="left")

    # --- 7. select and return final columns -------------------------------
    interaction_cols = [
        "userID", "movieID", "timestamp",
        "interaction_rating",
        "interaction_rating_vs_user_avg",
        "interaction_rating_vs_movie_avg",
        "interaction_movie_age_at_rating",
        "interaction_is_early_rater",
        "interaction_user_rating_sequence",
    ] + [c for c in df.columns if c.startswith("user_genre_affinity_")]

    return df[interaction_cols].reset_index(drop=True)


def _build_genre_affinity(
    user_ratedmovies_df: pd.DataFrame,
    movie_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute each user's average rating per genre.

    Returns a DataFrame with one row per userID and columns:
        user_genre_affinity_<GenreName>

    WHY: Genre affinity is the core of content-based filtering.  For SHAP,
    a feature like ``user_genre_affinity_Horror=3.2`` clearly explains
    that this user mildly likes Horror, which influenced the recommendation.
    """
    genre_cols = [f"genre_{g}" for g in ALL_GENRES]

    # Join ratings with genre flags
    ratings_with_genres = user_ratedmovies_df[["userID", "movieID", "rating"]].merge(
        movie_features[["movieID"] + genre_cols],
        on="movieID",
        how="left",
    )

    affinity_frames = []
    for genre in ALL_GENRES:
        col = f"genre_{genre}"
        if col not in ratings_with_genres.columns:
            continue
        # Use only ratings for movies that have this genre
        genre_ratings = ratings_with_genres[ratings_with_genres[col] == 1]
        user_genre_avg = (
            genre_ratings
            .groupby("userID")["rating"]
            .mean()
            .reset_index(name=f"user_genre_affinity_{genre}")
        )
        affinity_frames.append(user_genre_avg)

    if not affinity_frames:
        return pd.DataFrame({"userID": user_ratedmovies_df["userID"].unique()})

    # Merge all genre affinity columns together
    result = affinity_frames[0]
    for frame in affinity_frames[1:]:
        result = result.merge(frame, on="userID", how="outer")

    # Fill genres the user has never rated with the global average (3.5)
    for genre in ALL_GENRES:
        col = f"user_genre_affinity_{genre}"
        if col in result.columns:
            result[col] = result[col].fillna(3.5)

    return result


# ===========================================================================
# TOP-LEVEL ORCHESTRATOR
# ===========================================================================


def build_all_features(
    data: dict,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build all three feature groups from the raw data dictionary.

    Parameters
    ----------
    data : dict
        Output of load_data_tables(file_map) — keys are table names,
        values are DataFrames.

    Returns
    -------
    user_features        : pd.DataFrame  (one row per userID)
    movie_features       : pd.DataFrame  (one row per movieID)
    interaction_features : pd.DataFrame  (one row per userID-movieID rating)

    Example
    -------
    >>> user_feats, movie_feats, interact_feats = build_all_features(data)
    >>> print(user_feats.shape, movie_feats.shape, interact_feats.shape)
    """
    print("Building user features...")
    user_features = build_user_features(
        user_ratedmovies_df=data["user_ratedmovies"],
        user_taggedmovies_df=data["user_taggedmovies"],
    )
    print(f"  -> user_features: {user_features.shape}")

    print("Building movie features...")
    movie_features = build_movie_features(
        movies_df=data["movies"],
        movie_actors_df=data["movie_actors"],
        movie_genres_df=data["movie_genres"],
        movie_directors_df=data["movie_directors"],
        movie_countries_df=data["movie_countries"],
        movie_tags_df=data["movie_tags"],
        movie_locations_df=data["movie_locations"],
        user_ratedmovies_df=data["user_ratedmovies"],
    )
    print(f"  -> movie_features: {movie_features.shape}")

    print("Building interaction features...")
    interaction_features = build_interaction_features(
        user_ratedmovies_df=data["user_ratedmovies"],
        user_features=user_features,
        movie_features=movie_features,
        movies_df=data["movies"],
    )
    print(f"  -> interaction_features: {interaction_features.shape}")

    return user_features, movie_features, interaction_features


# ===========================================================================
# FEATURE REGISTRY
# ---------------------------------------------------------------------------
# A human-readable catalogue of all engineered features.
# Use this dict to power your MLflow / W&B logging:
#
#   mlflow.log_param("feature_group", FEATURE_REGISTRY[feat]["group"])
#
# and to generate SHAP plot axis labels:
#
#   shap_labels = {k: v["description"] for k, v in FEATURE_REGISTRY.items()}
# ===========================================================================

FEATURE_REGISTRY = {
    # --- USER ---
    "user_rating_count": {
        "group": "user",
        "dtype": "int",
        "description": "Total number of ratings by this user",
        "shap_label": "User # ratings",
    },
    "user_avg_rating": {
        "group": "user",
        "dtype": "float",
        "description": "Mean rating given by this user (leniency bias)",
        "shap_label": "User avg rating",
    },
    "user_rating_std": {
        "group": "user",
        "dtype": "float",
        "description": "Std dev of user ratings (taste consistency)",
        "shap_label": "User rating std",
    },
    "user_rating_range": {
        "group": "user",
        "dtype": "float",
        "description": "Max minus min rating used (scale breadth)",
        "shap_label": "User rating range",
    },
    "user_pct_high_ratings": {
        "group": "user",
        "dtype": "float",
        "description": "Share of ratings ≥ 4.0 (positivity tendency)",
        "shap_label": "User % high ratings",
    },
    "user_activity_span_days": {
        "group": "user",
        "dtype": "int",
        "description": "Days between first and last rating (tenure)",
        "shap_label": "User activity span (days)",
    },
    "user_days_since_last_rating": {
        "group": "user",
        "dtype": "int",
        "description": "Days since most recent rating (recency)",
        "shap_label": "Days since last rating",
    },
    "user_avg_rating_log_count": {
        "group": "user",
        "dtype": "float",
        "description": "avg_rating × log1p(count) — engagement × sentiment",
        "shap_label": "User engagement score",
    },
    "user_tag_event_count": {
        "group": "user",
        "dtype": "int",
        "description": "Total tagging events (implicit engagement)",
        "shap_label": "User tag events",
    },
    "user_tagging_intensity": {
        "group": "user",
        "dtype": "float",
        "description": "Tag events per unique movie tagged (depth of engagement)",
        "shap_label": "User tagging intensity",
    },
    # --- MOVIE ---
    "movie_year": {
        "group": "movie",
        "dtype": "int",
        "description": "Release year of the movie",
        "shap_label": "Movie release year",
    },
    "movie_age": {
        "group": "movie",
        "dtype": "int",
        "description": "Years since release (at dataset end date)",
        "shap_label": "Movie age (years)",
    },
    "movie_rating_count": {
        "group": "movie",
        "dtype": "int",
        "description": "Total ratings received (raw popularity)",
        "shap_label": "Movie # ratings",
    },
    "movie_log_rating_count": {
        "group": "movie",
        "dtype": "float",
        "description": "log1p of rating count (compressed popularity)",
        "shap_label": "Movie log-popularity",
    },
    "movie_avg_rating": {
        "group": "movie",
        "dtype": "float",
        "description": "Mean rating across all users",
        "shap_label": "Movie avg rating",
    },
    "movie_rating_std": {
        "group": "movie",
        "dtype": "float",
        "description": "Std of ratings (polarisation score)",
        "shap_label": "Movie rating polarisation",
    },
    "movie_has_rt_data": {
        "group": "movie",
        "dtype": "int",
        "description": "1 if Rotten Tomatoes data is available",
        "shap_label": "Has RT data",
    },
    "movie_rt_audience_rating": {
        "group": "movie",
        "dtype": "float",
        "description": "RT audience average rating",
        "shap_label": "RT audience rating",
    },
    "movie_genre_count": {
        "group": "movie",
        "dtype": "int",
        "description": "Number of genres assigned to this movie",
        "shap_label": "# genres",
    },
    "movie_cast_size_capped": {
        "group": "movie",
        "dtype": "int",
        "description": f"Cast size capped at {CAST_SIZE_CAP} (outlier-robust)",
        "shap_label": "Cast size",
    },
    "movie_director_experience": {
        "group": "movie",
        "dtype": "int",
        "description": "Number of movies the director has made",
        "shap_label": "Director experience",
    },
    "movie_unique_tag_count": {
        "group": "movie",
        "dtype": "int",
        "description": "Number of distinct tags assigned",
        "shap_label": "Movie tag richness",
    },
    "movie_log_tag_weight_sum": {
        "group": "movie",
        "dtype": "float",
        "description": "log1p of total tag assignment count",
        "shap_label": "Movie tag volume (log)",
    },
    "movie_location_depth": {
        "group": "movie",
        "dtype": "int",
        "description": "Number of non-null filming location levels (0-4)",
        "shap_label": "Location depth",
    },
    # --- INTERACTION ---
    "interaction_rating_vs_user_avg": {
        "group": "interaction",
        "dtype": "float",
        "description": "Rating minus user's personal mean (personal deviation)",
        "shap_label": "Rating vs user avg",
    },
    "interaction_rating_vs_movie_avg": {
        "group": "interaction",
        "dtype": "float",
        "description": "Rating minus movie's global mean (crowd deviation)",
        "shap_label": "Rating vs movie avg",
    },
    "interaction_movie_age_at_rating": {
        "group": "interaction",
        "dtype": "int",
        "description": "Movie age in years at the time of the rating",
        "shap_label": "Movie age at rating",
    },
    "interaction_is_early_rater": {
        "group": "interaction",
        "dtype": "int",
        "description": "1 if user rated within 1 year of release",
        "shap_label": "Early rater",
    },
    "interaction_user_rating_sequence": {
        "group": "interaction",
        "dtype": "int",
        "description": "Position of this rating in user's history (1 = first)",
        "shap_label": "User rating sequence pos.",
    },
}

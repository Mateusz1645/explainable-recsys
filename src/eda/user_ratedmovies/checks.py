from __future__ import annotations

import pandas as pd


def suspicious_ratings_report(user_ratedmovies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect basic quality issues in ``user_ratedmovies``."""
    required = {
        "userID",
        "movieID",
        "rating",
        "date_day",
        "date_month",
        "date_year",
        "date_hour",
        "date_minute",
        "date_second",
    }
    if not required.issubset(user_ratedmovies.columns):
        raise KeyError("user_ratedmovies missing required columns")

    df = user_ratedmovies.copy()
    rating = pd.to_numeric(df["rating"], errors="coerce")

    missing_rating = rating.isna()
    below_min = rating < 0.5
    above_max = rating > 5.0
    non_half_step = rating.notna() & ((rating * 2) % 1 != 0)
    duplicate_user_movie = df.duplicated(subset=["userID", "movieID"])

    suspicious_mask = missing_rating | below_min | above_max | non_half_step | duplicate_user_movie
    suspicious_rows = df.loc[suspicious_mask].copy()
    suspicious_rows["reason"] = ""
    suspicious_rows.loc[missing_rating.loc[suspicious_rows.index], "reason"] += "missing_rating;"
    suspicious_rows.loc[below_min.loc[suspicious_rows.index], "reason"] += "rating_below_0_5;"
    suspicious_rows.loc[above_max.loc[suspicious_rows.index], "reason"] += "rating_above_5;"
    suspicious_rows.loc[non_half_step.loc[suspicious_rows.index], "reason"] += "rating_not_half_step;"
    suspicious_rows.loc[duplicate_user_movie.loc[suspicious_rows.index], "reason"] += "duplicate_user_movie;"
    suspicious_rows["reason"] = suspicious_rows["reason"].str.strip(";")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "missing_rating",
                "rating_below_0_5",
                "rating_above_5",
                "rating_not_half_step",
                "duplicate_user_movie_pairs",
                "suspicious_rows_total",
            ],
            "value": [
                int(len(df)),
                int(missing_rating.sum()),
                int(below_min.sum()),
                int(above_max.sum()),
                int(non_half_step.sum()),
                int(duplicate_user_movie.sum()),
                int(suspicious_mask.sum()),
            ],
        }
    )
    return {"summary": summary, "suspicious_rows": suspicious_rows}


def rating_distribution_report(user_ratedmovies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize explicit rating value distribution."""
    if "rating" not in user_ratedmovies.columns:
        raise KeyError("user_ratedmovies must contain: rating")

    rating = pd.to_numeric(user_ratedmovies["rating"], errors="coerce").rename("rating")
    summary = pd.DataFrame(
        {
            "metric": ["count", "mean", "median", "p90", "p95", "p99", "min", "max"],
            "value": [
                int(rating.count()),
                round(float(rating.mean()), 3),
                round(float(rating.median()), 3),
                round(float(rating.quantile(0.90)), 3),
                round(float(rating.quantile(0.95)), 3),
                round(float(rating.quantile(0.99)), 3),
                float(rating.min()),
                float(rating.max()),
            ],
        }
    )
    counts = rating.value_counts().sort_index().rename("count").to_frame()
    return {"summary": summary, "distribution": rating.to_frame(), "value_counts": counts}


def ratings_time_report(user_ratedmovies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Build timestamp from split columns and summarize temporal coverage."""
    required = {"date_year", "date_month", "date_day", "date_hour", "date_minute", "date_second"}
    if not required.issubset(user_ratedmovies.columns):
        raise KeyError("user_ratedmovies missing date/time columns")

    event_time = pd.to_datetime(
        {
            "year": user_ratedmovies["date_year"],
            "month": user_ratedmovies["date_month"],
            "day": user_ratedmovies["date_day"],
            "hour": user_ratedmovies["date_hour"],
            "minute": user_ratedmovies["date_minute"],
            "second": user_ratedmovies["date_second"],
        },
        errors="coerce",
    )
    summary = pd.DataFrame(
        {
            "metric": ["rows_total", "missing_event_time", "min_event_time", "max_event_time"],
            "value": [
                int(len(user_ratedmovies)),
                int(event_time.isna().sum()),
                str(event_time.min()),
                str(event_time.max()),
            ],
        }
    )
    by_year = event_time.dt.year.value_counts().sort_index().rename("ratings").to_frame()
    return {"summary": summary, "distribution": event_time.rename("event_time").to_frame(), "by_year": by_year}


def ratings_activity_report(user_ratedmovies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize user and movie activity density in ratings data."""
    required = {"userID", "movieID"}
    if not required.issubset(user_ratedmovies.columns):
        raise KeyError("user_ratedmovies must contain: userID, movieID")

    users = user_ratedmovies["userID"].nunique()
    movies = user_ratedmovies["movieID"].nunique()
    interactions = len(user_ratedmovies)
    sparsity = 1 - (interactions / (users * movies)) if users and movies else 1.0

    ratings_per_user = user_ratedmovies.groupby("userID").size().rename("ratings_per_user")
    ratings_per_movie = user_ratedmovies.groupby("movieID").size().rename("ratings_per_movie")

    summary = pd.DataFrame(
        {
            "metric": [
                "users",
                "movies",
                "interactions",
                "sparsity",
                "densitsy",
            ],
            "value": [
                int(users),
                int(movies),
                int(interactions),
                round(float(sparsity), 6),
                round(float(1 - sparsity), 6),
            ],
        }
    )

    user_summary = pd.DataFrame(
        {
            "metric": ["mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                round(float(ratings_per_user.mean()), 3),
                round(float(ratings_per_user.median()), 3),
                round(float(ratings_per_user.quantile(0.90)), 3),
                round(float(ratings_per_user.quantile(0.95)), 3),
                round(float(ratings_per_user.quantile(0.99)), 3),
                int(ratings_per_user.max()),
            ],
        }
    )
    movie_summary = pd.DataFrame(
        {
            "metric": ["mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                round(float(ratings_per_movie.mean()), 3),
                round(float(ratings_per_movie.median()), 3),
                round(float(ratings_per_movie.quantile(0.90)), 3),
                round(float(ratings_per_movie.quantile(0.95)), 3),
                round(float(ratings_per_movie.quantile(0.99)), 3),
                int(ratings_per_movie.max()),
            ],
        }
    )
    return {
        "summary": summary,
        "ratings_per_user": ratings_per_user.to_frame(),
        "ratings_per_movie": ratings_per_movie.to_frame(),
        "ratings_per_user_summary": user_summary,
        "ratings_per_movie_summary": movie_summary,
    }


def ratings_coverage_report(user_ratedmovies: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Report movie ID coverage of ratings table against movies catalog."""
    if "movieID" not in user_ratedmovies.columns:
        raise KeyError("user_ratedmovies must contain: movieID")
    if "id" not in movies.columns:
        raise KeyError("movies must contain: id")

    movie_ids = set(movies["id"].unique())
    coverage_pct = round(user_ratedmovies["movieID"].isin(movie_ids).mean() * 100, 3)
    return pd.DataFrame(
        {
            "metric": ["rows_total", "unique_movies_in_ratings", "movie_id_coverage_pct"],
            "value": [int(len(user_ratedmovies)), int(user_ratedmovies["movieID"].nunique()), coverage_pct],
        }
    )

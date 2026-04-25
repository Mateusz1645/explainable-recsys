from __future__ import annotations

import pandas as pd


def suspicious_tag_events_report(user_taggedmovies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect structural issues in ``user_taggedmovies`` records."""
    required = {
        "userID",
        "movieID",
        "tagID",
        "date_day",
        "date_month",
        "date_year",
        "date_hour",
        "date_minute",
        "date_second",
    }
    if not required.issubset(user_taggedmovies.columns):
        raise KeyError("user_taggedmovies missing required columns")

    df = user_taggedmovies.copy()
    duplicate_triplet = df.duplicated(subset=["userID", "movieID", "tagID"])

    event_time = pd.to_datetime(
        {
            "year": df["date_year"],
            "month": df["date_month"],
            "day": df["date_day"],
            "hour": df["date_hour"],
            "minute": df["date_minute"],
            "second": df["date_second"],
        },
        errors="coerce",
    )
    invalid_time = event_time.isna()
    suspicious_mask = duplicate_triplet | invalid_time
    suspicious_rows = df.loc[suspicious_mask].copy()
    suspicious_rows["reason"] = ""
    suspicious_rows.loc[duplicate_triplet.loc[suspicious_rows.index], "reason"] += "duplicate_user_movie_tag;"
    suspicious_rows.loc[invalid_time.loc[suspicious_rows.index], "reason"] += "invalid_event_time;"
    suspicious_rows["reason"] = suspicious_rows["reason"].str.strip(";")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "duplicate_user_movie_tag",
                "invalid_event_time",
                "suspicious_rows_total",
            ],
            "value": [
                int(len(df)),
                int(duplicate_triplet.sum()),
                int(invalid_time.sum()),
                int(suspicious_mask.sum()),
            ],
        }
    )
    return {"summary": summary, "suspicious_rows": suspicious_rows}


def tagged_coverage_report(user_taggedmovies: pd.DataFrame, movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Report movie/tag ID coverage for tagging events."""
    if "movieID" not in user_taggedmovies.columns or "tagID" not in user_taggedmovies.columns:
        raise KeyError("user_taggedmovies must contain: movieID, tagID")
    if "id" not in movies.columns or "id" not in tags.columns:
        raise KeyError("movies and tags must contain: id")

    movie_ids = set(movies["id"].unique())
    tag_ids = set(tags["id"].unique())
    movie_cov = round(user_taggedmovies["movieID"].isin(movie_ids).mean() * 100, 3)
    tag_cov = round(user_taggedmovies["tagID"].isin(tag_ids).mean() * 100, 3)

    return pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "unique_users",
                "unique_movies",
                "unique_tags",
                "movie_id_coverage_pct",
                "tag_id_coverage_pct",
            ],
            "value": [
                int(len(user_taggedmovies)),
                int(user_taggedmovies["userID"].nunique()),
                int(user_taggedmovies["movieID"].nunique()),
                int(user_taggedmovies["tagID"].nunique()),
                movie_cov,
                tag_cov,
            ],
        }
    )


def tagged_time_report(user_taggedmovies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize temporal coverage of user tagging events."""
    required = {"date_year", "date_month", "date_day", "date_hour", "date_minute", "date_second"}
    if not required.issubset(user_taggedmovies.columns):
        raise KeyError("user_taggedmovies missing date/time columns")

    event_time = pd.to_datetime(
        {
            "year": user_taggedmovies["date_year"],
            "month": user_taggedmovies["date_month"],
            "day": user_taggedmovies["date_day"],
            "hour": user_taggedmovies["date_hour"],
            "minute": user_taggedmovies["date_minute"],
            "second": user_taggedmovies["date_second"],
        },
        errors="coerce",
    )
    summary = pd.DataFrame(
        {
            "metric": ["rows_total", "missing_event_time", "min_event_time", "max_event_time"],
            "value": [
                int(len(user_taggedmovies)),
                int(event_time.isna().sum()),
                str(event_time.min()),
                str(event_time.max()),
            ],
        }
    )
    by_year = event_time.dt.year.value_counts().sort_index().rename("tag_events").to_frame()
    return {"summary": summary, "distribution": event_time.rename("event_time").to_frame(), "by_year": by_year}


def tagged_activity_report(user_taggedmovies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize user/movie/tag activity distributions for tag events."""
    required = {"userID", "movieID", "tagID"}
    if not required.issubset(user_taggedmovies.columns):
        raise KeyError("user_taggedmovies must contain: userID, movieID, tagID")

    events_per_user = user_taggedmovies.groupby("userID").size().rename("tag_events_per_user")
    events_per_movie = user_taggedmovies.groupby("movieID").size().rename("tag_events_per_movie")
    events_per_tag = user_taggedmovies.groupby("tagID").size().rename("tag_event_uses_per_tag")

    summary = pd.DataFrame(
        {
            "metric": ["users", "movies", "tags", "events_total"],
            "value": [
                int(user_taggedmovies["userID"].nunique()),
                int(user_taggedmovies["movieID"].nunique()),
                int(user_taggedmovies["tagID"].nunique()),
                int(len(user_taggedmovies)),
            ],
        }
    )

    def describe_series(series: pd.Series) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "metric": ["mean", "median", "p90", "p95", "p99", "max"],
                "value": [
                    round(float(series.mean()), 3),
                    round(float(series.median()), 3),
                    round(float(series.quantile(0.90)), 3),
                    round(float(series.quantile(0.95)), 3),
                    round(float(series.quantile(0.99)), 3),
                    int(series.max()),
                ],
            }
        )

    return {
        "summary": summary,
        "events_per_user": events_per_user.to_frame(),
        "events_per_movie": events_per_movie.to_frame(),
        "events_per_tag": events_per_tag.to_frame(),
        "events_per_user_summary": describe_series(events_per_user),
        "events_per_movie_summary": describe_series(events_per_movie),
        "events_per_tag_summary": describe_series(events_per_tag),
    }

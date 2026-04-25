from __future__ import annotations

import pandas as pd


def suspicious_director_report(movie_directors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect potentially invalid director name entries in ``movie_directors``."""
    required = {"movieID", "directorID", "directorName"}
    if not required.issubset(movie_directors.columns):
        raise KeyError("movie_directors must contain: movieID, directorID, directorName")

    df = movie_directors.copy()
    name_as_str = df["directorName"].astype(str)

    missing_mask = df["directorName"].isna()
    empty_mask = name_as_str.str.strip().eq("")
    very_short_mask = name_as_str.str.strip().str.len().between(1, 2)
    numeric_only_mask = name_as_str.str.strip().str.fullmatch(r"\d+", na=False)
    suspicious_mask = missing_mask | empty_mask | very_short_mask | numeric_only_mask

    suspicious_rows = df.loc[suspicious_mask].copy()
    suspicious_rows["reason"] = ""
    suspicious_rows.loc[missing_mask.loc[suspicious_rows.index], "reason"] += "missing;"
    suspicious_rows.loc[empty_mask.loc[suspicious_rows.index], "reason"] += "empty_or_whitespace;"
    suspicious_rows.loc[very_short_mask.loc[suspicious_rows.index], "reason"] += "very_short;"
    suspicious_rows.loc[numeric_only_mask.loc[suspicious_rows.index], "reason"] += "numeric_only;"
    suspicious_rows["reason"] = suspicious_rows["reason"].str.strip(";")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "missing_director_name",
                "empty_or_whitespace_director_name",
                "very_short_director_name_len_1_2",
                "numeric_only_director_name",
                "suspicious_rows_total",
            ],
            "value": [
                int(len(df)),
                int(missing_mask.sum()),
                int(empty_mask.sum()),
                int(very_short_mask.sum()),
                int(numeric_only_mask.sum()),
                int(suspicious_mask.sum()),
            ],
        }
    )
    return {"summary": summary, "suspicious_rows": suspicious_rows}


def directors_per_movie_report(movie_directors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of directors assigned per movie."""
    required = {"movieID", "directorID"}
    if not required.issubset(movie_directors.columns):
        raise KeyError("movie_directors must contain: movieID, directorID")

    directors_per_movie = (
        movie_directors.drop_duplicates(subset=["movieID", "directorID"])
        .groupby("movieID")
        .size()
        .rename("directors_per_movie")
    )

    summary = pd.DataFrame(
        {
            "metric": ["movies_with_director", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(directors_per_movie.shape[0]),
                round(float(directors_per_movie.mean()), 3),
                round(float(directors_per_movie.median()), 3),
                round(float(directors_per_movie.quantile(0.90)), 3),
                round(float(directors_per_movie.quantile(0.95)), 3),
                round(float(directors_per_movie.quantile(0.99)), 3),
                int(directors_per_movie.max()),
            ],
        }
    )

    top_movies = directors_per_movie.sort_values(ascending=False).head(20).to_frame()
    return {
        "summary": summary,
        "distribution": directors_per_movie.to_frame(),
        "top_movies": top_movies,
    }


def movies_per_director_report(movie_directors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of movies per director."""
    required = {"movieID", "directorID", "directorName"}
    if not required.issubset(movie_directors.columns):
        raise KeyError("movie_directors must contain: movieID, directorID, directorName")

    dedup = movie_directors.drop_duplicates(subset=["movieID", "directorID"]).copy()
    movies_per_director = dedup.groupby("directorID").size().rename("movies_per_director")

    director_names = (
        dedup[["directorID", "directorName"]]
        .dropna()
        .drop_duplicates(subset=["directorID"])
        .set_index("directorID")
    )

    summary = pd.DataFrame(
        {
            "metric": ["directors_total", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(movies_per_director.shape[0]),
                round(float(movies_per_director.mean()), 3),
                round(float(movies_per_director.median()), 3),
                round(float(movies_per_director.quantile(0.90)), 3),
                round(float(movies_per_director.quantile(0.95)), 3),
                round(float(movies_per_director.quantile(0.99)), 3),
                int(movies_per_director.max()),
            ],
        }
    )

    top_directors = (
        movies_per_director
        .sort_values(ascending=False)
        .head(20)
        .to_frame()
        .join(director_names, how="left")
    )
    return {
        "summary": summary,
        "distribution": movies_per_director.to_frame(),
        "top_directors": top_directors,
    }


def director_coverage_report(movie_directors: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Report how many movies have at least one director assignment."""
    if "movieID" not in movie_directors.columns:
        raise KeyError("movie_directors must contain: movieID")
    if "id" not in movies.columns:
        raise KeyError("movies must contain: id")

    movies_total = int(movies["id"].nunique())
    movies_with_director = int(movie_directors["movieID"].nunique())
    coverage_pct = round((movies_with_director / movies_total) * 100, 3) if movies_total else 0.0

    return pd.DataFrame(
        {
            "metric": ["movies_total", "movies_with_director", "coverage_pct"],
            "value": [movies_total, movies_with_director, coverage_pct],
        }
    )

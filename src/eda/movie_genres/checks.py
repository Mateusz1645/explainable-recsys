from __future__ import annotations

import pandas as pd


def suspicious_genre_report(movie_genres: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect potentially invalid genre values in ``movie_genres``."""
    required = {"movieID", "genre"}
    if not required.issubset(movie_genres.columns):
        raise KeyError("movie_genres must contain: movieID, genre")

    df = movie_genres.copy()
    genre_as_str = df["genre"].astype(str)

    missing_mask = df["genre"].isna()
    empty_mask = genre_as_str.str.strip().eq("")
    one_char_mask = genre_as_str.str.strip().str.len().eq(1)
    numeric_only_mask = genre_as_str.str.strip().str.fullmatch(r"\d+", na=False)
    suspicious_mask = missing_mask | empty_mask | one_char_mask | numeric_only_mask

    suspicious_rows = df.loc[suspicious_mask].copy()
    suspicious_rows["reason"] = ""
    suspicious_rows.loc[missing_mask.loc[suspicious_rows.index], "reason"] += "missing;"
    suspicious_rows.loc[empty_mask.loc[suspicious_rows.index], "reason"] += "empty_or_whitespace;"
    suspicious_rows.loc[one_char_mask.loc[suspicious_rows.index], "reason"] += "one_character;"
    suspicious_rows.loc[numeric_only_mask.loc[suspicious_rows.index], "reason"] += "numeric_only;"
    suspicious_rows["reason"] = suspicious_rows["reason"].str.strip(";")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "missing_genre",
                "empty_or_whitespace_genre",
                "one_character_genre",
                "numeric_only_genre",
                "suspicious_rows_total",
            ],
            "value": [
                int(len(df)),
                int(missing_mask.sum()),
                int(empty_mask.sum()),
                int(one_char_mask.sum()),
                int(numeric_only_mask.sum()),
                int(suspicious_mask.sum()),
            ],
        }
    )
    return {"summary": summary, "suspicious_rows": suspicious_rows}


def genres_per_movie_report(movie_genres: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of genres assigned per movie."""
    required = {"movieID", "genre"}
    if not required.issubset(movie_genres.columns):
        raise KeyError("movie_genres must contain: movieID, genre")

    genres_per_movie = (
        movie_genres.drop_duplicates(subset=["movieID", "genre"])
        .groupby("movieID")
        .size()
        .rename("genres_per_movie")
    )

    summary = pd.DataFrame(
        {
            "metric": ["movies_with_genre", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(genres_per_movie.shape[0]),
                round(float(genres_per_movie.mean()), 3),
                round(float(genres_per_movie.median()), 3),
                round(float(genres_per_movie.quantile(0.90)), 3),
                round(float(genres_per_movie.quantile(0.95)), 3),
                round(float(genres_per_movie.quantile(0.99)), 3),
                int(genres_per_movie.max()),
            ],
        }
    )

    top_movies = genres_per_movie.sort_values(ascending=False).head(20).to_frame()
    return {
        "summary": summary,
        "distribution": genres_per_movie.to_frame(),
        "top_movies": top_movies,
    }


def movies_per_genre_report(movie_genres: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of movies assigned to each genre."""
    required = {"movieID", "genre"}
    if not required.issubset(movie_genres.columns):
        raise KeyError("movie_genres must contain: movieID, genre")

    movies_per_genre = (
        movie_genres.drop_duplicates(subset=["movieID", "genre"])
        .groupby("genre")
        .size()
        .rename("movies_per_genre")
        .sort_values(ascending=False)
    )

    summary = pd.DataFrame(
        {
            "metric": ["genres_total", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(movies_per_genre.shape[0]),
                round(float(movies_per_genre.mean()), 3),
                round(float(movies_per_genre.median()), 3),
                round(float(movies_per_genre.quantile(0.90)), 3),
                round(float(movies_per_genre.quantile(0.95)), 3),
                round(float(movies_per_genre.quantile(0.99)), 3),
                int(movies_per_genre.max()),
            ],
        }
    )

    top_genres = movies_per_genre.head(20).to_frame()
    return {
        "summary": summary,
        "distribution": movies_per_genre.to_frame(),
        "top_genres": top_genres,
    }


def genre_coverage_report(movie_genres: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Report how many movies have at least one genre assignment."""
    if "movieID" not in movie_genres.columns:
        raise KeyError("movie_genres must contain: movieID")
    if "id" not in movies.columns:
        raise KeyError("movies must contain: id")

    movies_total = int(movies["id"].nunique())
    movies_with_genre = int(movie_genres["movieID"].nunique())
    coverage_pct = round((movies_with_genre / movies_total) * 100, 3) if movies_total else 0.0

    return pd.DataFrame(
        {
            "metric": ["movies_total", "movies_with_genre", "coverage_pct"],
            "value": [movies_total, movies_with_genre, coverage_pct],
        }
    )

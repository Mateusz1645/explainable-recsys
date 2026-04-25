from __future__ import annotations

import pandas as pd


def suspicious_country_report(movie_countries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect potentially invalid country values in ``movie_countries``.

    Flags missing, empty/whitespace, numeric-only, or single-character values.
    Two-character country names/codes are not flagged by default because
    abbreviations such as ``UK`` are valid in this dataset.
    """
    required = {"movieID", "country"}
    if not required.issubset(movie_countries.columns):
        raise KeyError("movie_countries must contain: movieID, country")

    df = movie_countries.copy()
    country_as_str = df["country"].astype(str)

    missing_mask = df["country"].isna()
    empty_mask = country_as_str.str.strip().eq("")
    numeric_only_mask = country_as_str.str.strip().str.fullmatch(r"\d+", na=False)
    one_char_mask = country_as_str.str.strip().str.len().eq(1)

    suspicious_mask = missing_mask | empty_mask | numeric_only_mask | one_char_mask
    suspicious_rows = df.loc[suspicious_mask].copy()
    suspicious_rows["reason"] = ""
    suspicious_rows.loc[missing_mask.loc[suspicious_rows.index], "reason"] += "missing;"
    suspicious_rows.loc[empty_mask.loc[suspicious_rows.index], "reason"] += "empty_or_whitespace;"
    suspicious_rows.loc[numeric_only_mask.loc[suspicious_rows.index], "reason"] += "numeric_only;"
    suspicious_rows.loc[one_char_mask.loc[suspicious_rows.index], "reason"] += "one_character;"
    suspicious_rows["reason"] = suspicious_rows["reason"].str.strip(";")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "missing_country",
                "empty_or_whitespace_country",
                "numeric_only_country",
                "one_character_country",
                "suspicious_rows_total",
            ],
            "value": [
                int(len(df)),
                int(missing_mask.sum()),
                int(empty_mask.sum()),
                int(numeric_only_mask.sum()),
                int(one_char_mask.sum()),
                int(suspicious_mask.sum()),
            ],
        }
    )

    return {"summary": summary, "suspicious_rows": suspicious_rows}


def countries_per_movie_report(movie_countries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of countries assigned per movie."""
    required = {"movieID", "country"}
    if not required.issubset(movie_countries.columns):
        raise KeyError("movie_countries must contain: movieID, country")

    countries_per_movie = (
        movie_countries.drop_duplicates(subset=["movieID", "country"])
        .groupby("movieID")
        .size()
        .rename("countries_per_movie")
    )

    summary = pd.DataFrame(
        {
            "metric": ["movies_with_country", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(countries_per_movie.shape[0]),
                round(float(countries_per_movie.mean()), 3),
                round(float(countries_per_movie.median()), 3),
                round(float(countries_per_movie.quantile(0.90)), 3),
                round(float(countries_per_movie.quantile(0.95)), 3),
                round(float(countries_per_movie.quantile(0.99)), 3),
                int(countries_per_movie.max()),
            ],
        }
    )

    top_movies = countries_per_movie.sort_values(ascending=False).head(20).to_frame()
    return {
        "summary": summary,
        "distribution": countries_per_movie.to_frame(),
        "top_movies": top_movies,
    }


def movies_per_country_report(movie_countries: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of movies mapped to each country."""
    required = {"movieID", "country"}
    if not required.issubset(movie_countries.columns):
        raise KeyError("movie_countries must contain: movieID, country")

    movies_per_country = (
        movie_countries.drop_duplicates(subset=["movieID", "country"])
        .groupby("country")
        .size()
        .rename("movies_per_country")
        .sort_values(ascending=False)
    )

    summary = pd.DataFrame(
        {
            "metric": ["countries_total", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(movies_per_country.shape[0]),
                round(float(movies_per_country.mean()), 3),
                round(float(movies_per_country.median()), 3),
                round(float(movies_per_country.quantile(0.90)), 3),
                round(float(movies_per_country.quantile(0.95)), 3),
                round(float(movies_per_country.quantile(0.99)), 3),
                int(movies_per_country.max()),
            ],
        }
    )

    top_countries = movies_per_country.head(20).to_frame()
    return {
        "summary": summary,
        "distribution": movies_per_country.to_frame(),
        "top_countries": top_countries,
    }


def country_coverage_report(movie_countries: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Report how many movies have at least one country assignment."""
    required_movie_countries = {"movieID"}
    if not required_movie_countries.issubset(movie_countries.columns):
        raise KeyError("movie_countries must contain: movieID")
    if "id" not in movies.columns:
        raise KeyError("movies must contain: id")

    movies_total = int(movies["id"].nunique())
    movies_with_country = int(movie_countries["movieID"].nunique())
    coverage_pct = round((movies_with_country / movies_total) * 100, 3) if movies_total else 0.0

    return pd.DataFrame(
        {
            "metric": ["movies_total", "movies_with_country", "coverage_pct"],
            "value": [movies_total, movies_with_country, coverage_pct],
        }
    )

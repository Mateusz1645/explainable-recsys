from __future__ import annotations

import pandas as pd

LOCATION_COLS = ["location1", "location2", "location3", "location4"]


def suspicious_location_report(movie_locations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect potentially invalid values across location columns."""
    required = {"movieID", *LOCATION_COLS}
    if not required.issubset(movie_locations.columns):
        raise KeyError("movie_locations must contain movieID, location1-4")

    df = movie_locations.copy()

    long_df = df.melt(
        id_vars=["movieID"],
        value_vars=LOCATION_COLS,
        var_name="location_level",
        value_name="location_value",
    )
    value_str = long_df["location_value"].astype(str).str.strip()
    missing_mask = long_df["location_value"].isna()
    empty_mask = value_str.eq("")
    numeric_only_mask = value_str.str.fullmatch(r"\d+", na=False)
    one_char_mask = value_str.str.len().eq(1)
    suspicious_mask = missing_mask | empty_mask | numeric_only_mask | one_char_mask

    suspicious_rows = long_df.loc[suspicious_mask].copy()
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
                "missing_location_values",
                "empty_or_whitespace_values",
                "numeric_only_values",
                "one_character_values",
                "suspicious_values_total",
            ],
            "value": [
                int(len(long_df)),
                int(missing_mask.sum()),
                int(empty_mask.sum()),
                int(numeric_only_mask.sum()),
                int(one_char_mask.sum()),
                int(suspicious_mask.sum()),
            ],
        }
    )
    return {"summary": summary, "suspicious_rows": suspicious_rows}


def rows_per_movie_location_report(movie_locations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize how many location rows are available per movie."""
    if "movieID" not in movie_locations.columns:
        raise KeyError("movie_locations must contain: movieID")

    rows_per_movie = movie_locations.groupby("movieID").size().rename("rows_per_movie")
    summary = pd.DataFrame(
        {
            "metric": ["movies_with_locations", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(rows_per_movie.shape[0]),
                round(float(rows_per_movie.mean()), 3),
                round(float(rows_per_movie.median()), 3),
                round(float(rows_per_movie.quantile(0.90)), 3),
                round(float(rows_per_movie.quantile(0.95)), 3),
                round(float(rows_per_movie.quantile(0.99)), 3),
                int(rows_per_movie.max()),
            ],
        }
    )
    top_movies = rows_per_movie.sort_values(ascending=False).head(20).to_frame()
    return {"summary": summary, "distribution": rows_per_movie.to_frame(), "top_movies": top_movies}


def location_depth_report(movie_locations: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of non-null location levels per row."""
    required = set(LOCATION_COLS)
    if not required.issubset(movie_locations.columns):
        raise KeyError("movie_locations must contain location1-4")

    depth = (~movie_locations[LOCATION_COLS].isna()).sum(axis=1).rename("non_column_location")
    depth = depth[depth > 0]
    summary = pd.DataFrame(
        {
            "metric": ["rows_total", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(depth.shape[0]),
                round(float(depth.mean()), 3),
                round(float(depth.median()), 3),
                round(float(depth.quantile(0.90)), 3),
                round(float(depth.quantile(0.95)), 3),
                round(float(depth.quantile(0.99)), 3),
                int(depth.max()),
            ],
        }
    )
    depth_counts = depth.value_counts().sort_index().rename("rows").to_frame()
    return {"summary": summary, "distribution": depth.to_frame(), "depth_counts": depth_counts}


def location_coverage_report(movie_locations: pd.DataFrame, movies: pd.DataFrame) -> pd.DataFrame:
    """Report movie-level coverage of location metadata."""
    if "movieID" not in movie_locations.columns:
        raise KeyError("movie_locations must contain: movieID")
    if "id" not in movies.columns:
        raise KeyError("movies must contain: id")

    movies_total = int(movies["id"].nunique())
    movies_with_locations = int(movie_locations["movieID"].nunique())
    coverage_pct = round((movies_with_locations / movies_total) * 100, 3) if movies_total else 0.0
    return pd.DataFrame(
        {
            "metric": ["movies_total", "movies_with_locations", "coverage_pct"],
            "value": [movies_total, movies_with_locations, coverage_pct],
        }
    )

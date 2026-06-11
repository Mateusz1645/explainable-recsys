import numpy as np
import pandas as pd


def quality_report_global(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Generate table-level quality summary for all loaded datasets.

    The report includes row/column counts, total missing cells, and duplicate
    row counts for each table.
    """
    rows = []
    for name, df in data.items():
        rows.append(
            {
                "table": name,
                "rows": int(df.shape[0]),
                "cols": int(df.shape[1]),
                "missing_cells": int(df.isna().sum().sum()),
                "duplicate_rows": int(df.duplicated().sum()),
            }
        )
    return pd.DataFrame(rows).sort_values("table").reset_index(drop=True)


def integrity_report(data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Run cross-table referential integrity checks.

    Returns a DataFrame with check names and violation counts for key foreign
    key-like relations between raw tables.
    """
    movies = data["movies"]
    tags = data["tags"]
    movie_actors = data["movie_actors"]
    movie_countries = data["movie_countries"]
    movie_directors = data["movie_directors"]
    user_ratedmovies = data["user_ratedmovies"]
    movie_genres = data["movie_genres"]
    movie_locations = data["movie_locations"]
    user_taggedmovies = data["user_taggedmovies"]
    movie_tags = data["movie_tags"]

    movie_ids = set(movies["id"].unique())
    tag_ids = set(tags["id"].unique())

    return pd.DataFrame(
        {
            "check": [
                "movie_actors.movieID in movies.id",
                "movie_countries.movieID in movies.id",
                "movie_directors.movieID in movies.id",
                "user_ratedmovies.movieID in movies.id",
                "movie_genres.movieID in movies.id",
                "movie_locations.movieID in movies.id",
                "user_taggedmovies.movieID in movies.id",
                "user_taggedmovies.tagID in tags.id",
                "movie_tags.movieID in movies.id",
                "movie_tags.tagID in tags.id",
            ],
            "violations": [
                int((~movie_actors["movieID"].isin(movie_ids)).sum()),
                int((~movie_countries["movieID"].isin(movie_ids)).sum()),
                int((~movie_directors["movieID"].isin(movie_ids)).sum()),
                int((~user_ratedmovies["movieID"].isin(movie_ids)).sum()),
                int((~movie_genres["movieID"].isin(movie_ids)).sum()),
                int((~movie_locations["movieID"].isin(movie_ids)).sum()),
                int((~user_taggedmovies["movieID"].isin(movie_ids)).sum()),
                int((~user_taggedmovies["tagID"].isin(tag_ids)).sum()),
                int((~movie_tags["movieID"].isin(movie_ids)).sum()),
                int((~movie_tags["tagID"].isin(tag_ids)).sum()),
            ],
        }
    )


def dataframe_overview(df: pd.DataFrame) -> pd.DataFrame:
    """Create a column-level profile for a single DataFrame.

    For each column, this function reports dtype, missingness, cardinality,
    and basic numeric statistics (excluding ID-like fields).
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns

    ignore_stats_cols = {"id", "movieID", "userID", "actorID", "tagID"}

    rows = []

    for col in df.columns:
        series = df[col]

        is_id_like = col in ignore_stats_cols or col.lower().endswith("id")

        if col in numeric_cols and not is_id_like:
            col_type = "numeric"
            mean = series.mean()
            std = series.std()
            min_val = series.min()
            max_val = series.max()
        else:
            col_type = "non-numeric" if col not in numeric_cols else "id-like"
            mean = None
            std = None
            min_val = None
            max_val = None

        rows.append(
            {
                "column": col,
                "dtype": str(series.dtype),
                "type_group": col_type,
                "missing_count": int(series.isna().sum()),
                "missing_pct": round(series.isna().mean(), 4),
                "n_unique": int(series.nunique(dropna=True)),
                "mean": mean,
                "std": std,
                "min": min_val,
                "max": max_val,
            }
        )

    return pd.DataFrame(rows).sort_values("missing_count", ascending=False).reset_index(drop=True)

from __future__ import annotations

import pandas as pd


def suspicious_movie_tags_report(movie_tags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect basic quality issues in ``movie_tags``."""
    required = {"movieID", "tagID", "tagWeight"}
    if not required.issubset(movie_tags.columns):
        raise KeyError("movie_tags must contain: movieID, tagID, tagWeight")

    df = movie_tags.copy()
    weight = pd.to_numeric(df["tagWeight"], errors="coerce")

    missing_weight_mask = weight.isna()
    non_positive_weight_mask = weight <= 0
    non_integer_weight_mask = weight.notna() & (weight % 1 != 0)
    suspicious_mask = missing_weight_mask | non_positive_weight_mask | non_integer_weight_mask

    suspicious_rows = df.loc[suspicious_mask].copy()
    suspicious_rows["reason"] = ""
    suspicious_rows.loc[missing_weight_mask.loc[suspicious_rows.index], "reason"] += "missing_weight;"
    suspicious_rows.loc[non_positive_weight_mask.loc[suspicious_rows.index], "reason"] += "non_positive_weight;"
    suspicious_rows.loc[non_integer_weight_mask.loc[suspicious_rows.index], "reason"] += "non_integer_weight;"
    suspicious_rows["reason"] = suspicious_rows["reason"].str.strip(";")

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "missing_tag_weight",
                "non_positive_tag_weight",
                "non_integer_tag_weight",
                "suspicious_rows_total",
                "duplicate_movie_tag_pairs",
            ],
            "value": [
                int(len(df)),
                int(missing_weight_mask.sum()),
                int(non_positive_weight_mask.sum()),
                int(non_integer_weight_mask.sum()),
                int(suspicious_mask.sum()),
                int(df.duplicated(subset=["movieID", "tagID"]).sum()),
            ],
        }
    )
    return {"summary": summary, "suspicious_rows": suspicious_rows}


def movie_tags_coverage_report(movie_tags: pd.DataFrame, movies: pd.DataFrame, tags: pd.DataFrame) -> pd.DataFrame:
    """Report referential coverage of ``movieID`` and ``tagID`` in ``movie_tags``."""
    if "movieID" not in movie_tags.columns:
        raise KeyError("movie_tags must contain: movieID")
    if "tagID" not in movie_tags.columns:
        raise KeyError("movie_tags must contain: tagID")
    if "id" not in movies.columns:
        raise KeyError("movies must contain: id")
    if "id" not in tags.columns:
        raise KeyError("tags must contain: id")

    movie_ids = set(movies["id"].unique())
    tag_ids = set(tags["id"].unique())

    movie_cov = round(movie_tags["movieID"].isin(movie_ids).mean() * 100, 3)
    tag_cov = round(movie_tags["tagID"].isin(tag_ids).mean() * 100, 3)

    return pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "unique_movies_in_movie_tags",
                "unique_tags_in_movie_tags",
                "movie_id_coverage_pct",
                "tag_id_coverage_pct",
            ],
            "value": [
                int(len(movie_tags)),
                int(movie_tags["movieID"].nunique()),
                int(movie_tags["tagID"].nunique()),
                movie_cov,
                tag_cov,
            ],
        }
    )


def tag_weight_report(movie_tags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize distribution of ``tagWeight`` values."""
    if "tagWeight" not in movie_tags.columns:
        raise KeyError("movie_tags must contain: tagWeight")

    weight = pd.to_numeric(movie_tags["tagWeight"], errors="coerce").rename("tag_weight")
    summary = pd.DataFrame(
        {
            "metric": ["count", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(weight.count()),
                round(float(weight.mean()), 3),
                round(float(weight.median()), 3),
                round(float(weight.quantile(0.90)), 3),
                round(float(weight.quantile(0.95)), 3),
                round(float(weight.quantile(0.99)), 3),
                int(weight.max()),
            ],
        }
    )
    return {"summary": summary, "distribution": weight.to_frame()}


def tags_per_movie_report(movie_tags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of tag assignments per movie."""
    required = {"movieID", "tagID"}
    if not required.issubset(movie_tags.columns):
        raise KeyError("movie_tags must contain: movieID, tagID")

    tags_per_movie = movie_tags.groupby("movieID").size().rename("tags_per_movie")
    summary = pd.DataFrame(
        {
            "metric": ["movies_with_tags", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(tags_per_movie.shape[0]),
                round(float(tags_per_movie.mean()), 3),
                round(float(tags_per_movie.median()), 3),
                round(float(tags_per_movie.quantile(0.90)), 3),
                round(float(tags_per_movie.quantile(0.95)), 3),
                round(float(tags_per_movie.quantile(0.99)), 3),
                int(tags_per_movie.max()),
            ],
        }
    )
    top_movies = tags_per_movie.sort_values(ascending=False).head(20).to_frame()
    return {"summary": summary, "distribution": tags_per_movie.to_frame(), "top_movies": top_movies}


def movies_per_tag_report(movie_tags: pd.DataFrame, tags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize number of movies assigned to each tag."""
    if "tagID" not in movie_tags.columns:
        raise KeyError("movie_tags must contain: tagID")
    if not {"id", "value"}.issubset(tags.columns):
        raise KeyError("tags must contain: id, value")

    movies_per_tag = movie_tags.groupby("tagID").size().rename("movies_per_tag").sort_values(ascending=False)
    tag_names = tags.set_index("id")["value"].rename("tag_value")
    top_tags = movies_per_tag.head(20).to_frame().join(tag_names, how="left")

    summary = pd.DataFrame(
        {
            "metric": ["used_tags_total", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(movies_per_tag.shape[0]),
                round(float(movies_per_tag.mean()), 3),
                round(float(movies_per_tag.median()), 3),
                round(float(movies_per_tag.quantile(0.90)), 3),
                round(float(movies_per_tag.quantile(0.95)), 3),
                round(float(movies_per_tag.quantile(0.99)), 3),
                int(movies_per_tag.max()),
            ],
        }
    )
    return {"summary": summary, "distribution": movies_per_tag.to_frame(), "top_tags": top_tags}

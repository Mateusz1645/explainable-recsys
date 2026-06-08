from __future__ import annotations

import pandas as pd


def suspicious_tag_report(tags: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Detect potentially invalid values in the ``tags`` lookup table."""
    required = {"id", "value"}
    if not required.issubset(tags.columns):
        raise KeyError("tags must contain: id, value")

    df = tags.copy()
    value_str = df["value"].astype(str).str.strip()

    missing_mask = df["value"].isna()
    empty_mask = value_str.eq("")
    one_char_mask = value_str.str.len().eq(1)
    numeric_only_mask = value_str.str.fullmatch(r"\d+", na=False)
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
                "missing_tag_value",
                "empty_or_whitespace_tag_value",
                "one_character_tag_value",
                "numeric_only_tag_value",
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


def tags_coverage_report(tags: pd.DataFrame, movie_tags: pd.DataFrame, user_taggedmovies: pd.DataFrame) -> pd.DataFrame:
    """Report how well tag IDs used in interaction tables are covered by ``tags``."""
    if "id" not in tags.columns:
        raise KeyError("tags must contain: id")
    if "tagID" not in movie_tags.columns:
        raise KeyError("movie_tags must contain: tagID")
    if "tagID" not in user_taggedmovies.columns:
        raise KeyError("user_taggedmovies must contain: tagID")

    tags_ids = set(tags["id"].unique())
    movie_tag_ids = set(movie_tags["tagID"].unique())
    user_tag_ids = set(user_taggedmovies["tagID"].unique())
    used_ids = movie_tag_ids | user_tag_ids

    movie_coverage = round((len(movie_tag_ids & tags_ids) / len(movie_tag_ids)) * 100, 3) if movie_tag_ids else 0.0
    user_coverage = round((len(user_tag_ids & tags_ids) / len(user_tag_ids)) * 100, 3) if user_tag_ids else 0.0

    return pd.DataFrame(
        {
            "metric": [
                "tags_total",
                "movie_tag_ids_total",
                "user_tag_ids_total",
                "movie_tag_coverage_pct",
                "user_tag_coverage_pct",
                "unused_tags_in_lookup",
            ],
            "value": [
                int(len(tags_ids)),
                int(len(movie_tag_ids)),
                int(len(user_tag_ids)),
                movie_coverage,
                user_coverage,
                int(len(tags_ids - used_ids)),
            ],
        }
    )


def tag_usage_report(
    tags: pd.DataFrame, movie_tags: pd.DataFrame, user_taggedmovies: pd.DataFrame
) -> dict[str, pd.DataFrame]:
    """Summarize total tag usage frequency across movie and user tagging tables."""
    if "id" not in tags.columns:
        raise KeyError("tags must contain: id")
    if "tagID" not in movie_tags.columns:
        raise KeyError("movie_tags must contain: tagID")
    if "tagID" not in user_taggedmovies.columns:
        raise KeyError("user_taggedmovies must contain: tagID")

    movie_counts = movie_tags.groupby("tagID").size().rename("movie_tag_count")
    user_counts = user_taggedmovies.groupby("tagID").size().rename("user_tag_count")

    usage = movie_counts.to_frame().join(user_counts.to_frame(), how="outer").fillna(0)
    usage["movie_tag_count"] = usage["movie_tag_count"].astype(int)
    usage["user_tag_count"] = usage["user_tag_count"].astype(int)
    usage["total_usage"] = usage["movie_tag_count"] + usage["user_tag_count"]

    label_map = tags.set_index("id")["value"]
    usage = usage.join(label_map.rename("tag_value"), how="left")

    summary = pd.DataFrame(
        {
            "metric": ["used_tags_total", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(usage.shape[0]),
                round(float(usage["total_usage"].mean()), 3),
                round(float(usage["total_usage"].median()), 3),
                round(float(usage["total_usage"].quantile(0.90)), 3),
                round(float(usage["total_usage"].quantile(0.95)), 3),
                round(float(usage["total_usage"].quantile(0.99)), 3),
                int(usage["total_usage"].max()),
            ],
        }
    )

    top_tags = usage.sort_values("total_usage", ascending=False).head(20)
    return {
        "summary": summary,
        "distribution": usage[["total_usage"]].copy(),
        "top_tags": top_tags,
    }

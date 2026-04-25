from __future__ import annotations

import pandas as pd


def suspicious_actor_report(movie_actors: pd.DataFrame) -> dict[str, pd.DataFrame | int]:
    """Detect potentially invalid actor name entries in ``movie_actors``.

    Flags names that are missing, empty/whitespace, very short (1-2 chars), or
    numeric-only, and returns both summary counts and detailed rows.
    """
    df = movie_actors.copy()
    name_col = "actorName"

    if name_col not in df.columns:
        raise KeyError("movie_actors must contain 'actorName'")

    name_as_str = df[name_col].astype(str)
    missing_mask = df[name_col].isna()
    empty_mask = name_as_str.str.strip().eq("")
    very_short_mask = name_as_str.str.strip().str.len().between(1, 2)
    numeric_only_mask = name_as_str.str.strip().str.fullmatch(r"\d+", na=False)

    suspicious = df[missing_mask | empty_mask | very_short_mask | numeric_only_mask].copy()
    suspicious["reason"] = ""
    suspicious.loc[missing_mask.loc[suspicious.index], "reason"] += "missing;"
    suspicious.loc[empty_mask.loc[suspicious.index], "reason"] += "empty_or_whitespace;"
    suspicious.loc[very_short_mask.loc[suspicious.index], "reason"] += "very_short;"
    suspicious.loc[numeric_only_mask.loc[suspicious.index], "reason"] += "numeric_only;"
    suspicious["reason"] = suspicious["reason"].str.strip(";")

    return {
        "summary": pd.DataFrame(
            {
                "metric": [
                    "rows_total",
                    "missing_actor_name",
                    "empty_or_whitespace_actor_name",
                    "very_short_actor_name_len_1_2",
                    "numeric_only_actor_name",
                    "suspicious_rows_total",
                ],
                "value": [
                    int(len(df)),
                    int(missing_mask.sum()),
                    int(empty_mask.sum()),
                    int(very_short_mask.sum()),
                    int(numeric_only_mask.sum()),
                    int(len(suspicious)),
                ],
            }
        ),
        "suspicious_rows": suspicious,
    }


def actors_per_movie_report(movie_actors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize cast size distribution (number of actors per movie)."""
    required = {"movieID", "actorID"}
    if not required.issubset(movie_actors.columns):
        raise KeyError("movie_actors must contain: movieID, actorID")

    actors_per_movie = (
        movie_actors.drop_duplicates(subset=["movieID", "actorID"])
        .groupby("movieID")
        .size()
        .rename("actors_per_movie")
    )

    summary = pd.DataFrame(
        {
            "metric": ["movies_with_cast", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(actors_per_movie.shape[0]),
                round(float(actors_per_movie.mean()), 3),
                round(float(actors_per_movie.median()), 3),
                round(float(actors_per_movie.quantile(0.90)), 3),
                round(float(actors_per_movie.quantile(0.95)), 3),
                round(float(actors_per_movie.quantile(0.99)), 3),
                int(actors_per_movie.max()),
            ],
        }
    )

    top_movies = actors_per_movie.sort_values(ascending=False).head(20).to_frame()
    results = {
        "summary": summary,
        "distribution": actors_per_movie.to_frame(),
        "top_movies": top_movies
        }

    return results


def movies_per_actor_report(movie_actors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize career breadth distribution (number of movies per actor)."""
    required = {"movieID", "actorID", "actorName"}
    if not required.issubset(movie_actors.columns):
        raise KeyError("movie_actors must contain: movieID, actorID, actorName")

    dedup = movie_actors.drop_duplicates(subset=["movieID", "actorID"]).copy()
    movies_per_actor = dedup.groupby("actorID").size().rename("movies_per_actor")

    actor_names = (
        dedup[["actorID", "actorName"]]
        .dropna()
        .drop_duplicates(subset=["actorID"])
        .set_index("actorID")
    )

    summary = pd.DataFrame(
        {
            "metric": ["actors_total", "mean", "median", "p90", "p95", "p99", "max"],
            "value": [
                int(movies_per_actor.shape[0]),
                round(float(movies_per_actor.mean()), 3),
                round(float(movies_per_actor.median()), 3),
                round(float(movies_per_actor.quantile(0.90)), 3),
                round(float(movies_per_actor.quantile(0.95)), 3),
                round(float(movies_per_actor.quantile(0.99)), 3),
                int(movies_per_actor.max()),
            ],
        }
    )

    top_actors = (
        movies_per_actor
        .sort_values(ascending=False)
        .head(20)
        .to_frame()
        .join(actor_names, how="left")
    )

    results = {
        "summary": summary,
        "distribution": movies_per_actor.to_frame(),
        "top_actors": top_actors
        }

    return results


def ranking_imdb_order_report(movie_actors: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Assess quality of IMDb cast-order ranking in ``movie_actors``.

    Checks missing/non-positive/non-integer ranks, duplicate ranks within a
    movie, and whether ranking sequences are contiguous from 1.
    """
    required = {"movieID", "ranking"}
    if not required.issubset(movie_actors.columns):
        raise KeyError("movie_actors must contain: movieID, ranking")

    df = movie_actors.copy()
    ranking_num = pd.to_numeric(df["ranking"], errors="coerce")

    missing_ranking = ranking_num.isna()
    non_positive_ranking = ranking_num <= 0
    non_integer_ranking = ranking_num.notna() & (ranking_num % 1 != 0)

    per_movie_rank_dups = (
        df.assign(_ranking_num=ranking_num)
        .dropna(subset=["_ranking_num"])
        .duplicated(subset=["movieID", "_ranking_num"])
    )

    # Optional strictness: ranking sequence should start at 1 and be contiguous.
    rank_clean = df.assign(_ranking_num=ranking_num).dropna(subset=["_ranking_num"]).copy()
    rank_clean["_ranking_num"] = rank_clean["_ranking_num"].astype(int)

    by_movie = rank_clean.groupby("movieID")["_ranking_num"]
    min_rank = by_movie.min()
    n_unique = by_movie.nunique()
    max_rank = by_movie.max()

    non_contiguous_movie_ids = min_rank[
        (min_rank != 1) | ((max_rank - min_rank + 1) != n_unique)
    ].index

    summary = pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "missing_ranking",
                "non_positive_ranking",
                "non_integer_ranking",
                "duplicate_ranking_within_movie_rows",
                "movies_with_non_contiguous_ranking",
            ],
            "value": [
                int(len(df)),
                int(missing_ranking.sum()),
                int(non_positive_ranking.sum()),
                int(non_integer_ranking.sum()),
                int(per_movie_rank_dups.sum()),
                int(len(non_contiguous_movie_ids)),
            ],
        }
    )

    duplicate_ranking_rows = (
        df.assign(ranking_num=ranking_num)
        .loc[lambda x: x["ranking_num"].notna()]
        .loc[lambda x: x.duplicated(subset=["movieID", "ranking_num"], keep=False)]
        .sort_values(by=["movieID", "ranking_num"])  # type: ignore[arg-type]
    )

    # DataFrame with raw ranking values ​​-> to histogram
    ranking_distribution = pd.DataFrame({"ranking": ranking_num}).dropna()

    return {
        "summary": summary,
        "distribution": ranking_distribution,
        "duplicate_ranking_rows": duplicate_ranking_rows,
        "non_contiguous_movies": pd.DataFrame({"movieID": non_contiguous_movie_ids}),
    }

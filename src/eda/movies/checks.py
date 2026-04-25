from __future__ import annotations

import pandas as pd


def movies_missingness_report(movies: pd.DataFrame) -> pd.DataFrame:
    """Return column-level missingness report for ``movies`` table."""
    missing = movies.isna().sum().rename("missing_count").to_frame()
    missing["missing_pct"] = (missing["missing_count"] / len(movies)).round(4)
    return missing.sort_values("missing_count", ascending=False)


def movies_identifier_report(movies: pd.DataFrame) -> pd.DataFrame:
    """Validate uniqueness of core identifiers and common duplicate patterns."""
    required = {"id", "title", "year"}
    if not required.issubset(movies.columns):
        raise KeyError("movies must contain: id, title, year")

    return pd.DataFrame(
        {
            "metric": [
                "rows_total",
                "duplicate_full_rows",
                "duplicate_id",
                "duplicate_title_year",
                "unique_movie_ids",
            ],
            "value": [
                int(len(movies)),
                int(movies.duplicated().sum()),
                int(movies.duplicated(subset=["id"]).sum()),
                int(movies.duplicated(subset=["title", "year"]).sum()),
                int(movies["id"].nunique()),
            ],
        }
    )


def movies_year_report(movies: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Summarize year distribution and detect out-of-range values."""
    if "year" not in movies.columns:
        raise KeyError("movies must contain: year")

    year = pd.to_numeric(movies["year"], errors="coerce").rename("year")
    summary = pd.DataFrame(
        {
            "metric": [
                "count",
                "missing",
                "min",
                "median",
                "p90",
                "p99",
                "max",
                "year_le_1800",
                "year_gt_2026",
            ],
            "value": [
                int(year.count()),
                int(year.isna().sum()),
                int(year.min()) if year.notna().any() else None,
                float(year.median()) if year.notna().any() else None,
                float(year.quantile(0.90)) if year.notna().any() else None,
                float(year.quantile(0.99)) if year.notna().any() else None,
                int(year.max()) if year.notna().any() else None,
                int((year <= 1800).sum()),
                int((year > 2026).sum()),
            ],
        }
    )
    by_year = year.value_counts().sort_index().rename("movies").to_frame()
    return {"summary": summary, "distribution": year.to_frame(), "by_year": by_year}


def movies_text_quality_report(movies: pd.DataFrame) -> pd.DataFrame:
    """Check basic text quality signals in title and URL-like columns."""
    required = {"title", "imdbPictureURL"}
    if not required.issubset(movies.columns):
        raise KeyError("movies must contain: title, imdbPictureURL")

    title = movies["title"].astype(str).str.strip()
    poster = movies["imdbPictureURL"].astype(str).str.strip()

    return pd.DataFrame(
        {
            "metric": [
                "title_missing",
                "title_empty",
                "title_one_character",
                "imdbPictureURL_missing",
                "imdbPictureURL_empty",
            ],
            "value": [
                int(movies["title"].isna().sum()),
                int(title.eq("").sum()),
                int(title.str.len().eq(1).sum()),
                int(movies["imdbPictureURL"].isna().sum()),
                int(poster.eq("").sum()),
            ],
        }
    )


def rotten_tomatoes_coverage_report(movies: pd.DataFrame) -> pd.DataFrame:
    """Summarize completeness of Rotten Tomatoes related fields."""
    rt_cols = [col for col in movies.columns if col.startswith("rt")]
    if not rt_cols:
        raise KeyError("movies does not contain Rotten Tomatoes columns")

    rows = []
    for col in rt_cols:
        missing_count = int(movies[col].isna().sum())
        rows.append(
            {
                "column": col,
                "missing_count": missing_count,
                "missing_pct": round(missing_count / len(movies), 4),
            }
        )
    return pd.DataFrame(rows).sort_values("missing_count", ascending=False).reset_index(drop=True)

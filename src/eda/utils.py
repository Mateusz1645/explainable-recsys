import pandas as pd


def build_event_time(user_ratedmovies: pd.DataFrame) -> pd.Series:
    """Construct a timestamp series from split date/time columns.

    Parameters
    ----------
    user_ratedmovies : pd.DataFrame
        DataFrame containing ``date_year``, ``date_month``, ``date_day``,
        ``date_hour``, ``date_minute``, and ``date_second`` columns.

    Returns
    -------
    pd.Series
        Parsed datetime values. Invalid combinations are coerced to ``NaT``.
    """
    return pd.to_datetime(
        {
            "year": user_ratedmovies["date_year"],
            "month": user_ratedmovies["date_month"],
            "day": user_ratedmovies["date_day"],
            "hour": user_ratedmovies["date_hour"],
            "minute": user_ratedmovies["date_minute"],
            "second": user_ratedmovies["date_second"],
        },
        errors="coerce",
    )

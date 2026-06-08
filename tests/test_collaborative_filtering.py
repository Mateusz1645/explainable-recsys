import pandas as pd
import pytest

from src.models.collaborative_filtering import (
    SGDMatrixFactorization,
    SVDCollaborativeFiltering,
)


def _interactions():
    return pd.DataFrame(
        {
            "userID": [1, 1, 2, 2, 3, 3],
            "movieID": [10, 11, 10, 12, 11, 12],
            "interaction_rating": [4.0, 5.0, 3.0, 4.5, 2.5, 4.0],
        }
    )


def _metadata():
    return pd.DataFrame(
        {
            "movieID": [10, 11, 12],
            "title": ["M1", "M2", "M3"],
        }
    )


def test_svd_fit_predict_and_recommend():
    model = SVDCollaborativeFiltering(n_factors=1, random_state=42)
    model.fit(_interactions())

    pred = model.predict_rating(1, 12)
    recs = model.recommend(1, _metadata(), top_n=1)

    assert isinstance(pred, float)
    assert len(recs) == 1
    assert list(recs.columns) == ["movieID", "title", "predicted_rating"]


def test_svd_predict_rating_returns_nan_for_unknown_ids():
    model = SVDCollaborativeFiltering(n_factors=1, random_state=42)
    model.fit(_interactions())

    assert pd.isna(model.predict_rating(999, 12))
    assert pd.isna(model.predict_rating(1, 999))


def test_svd_recommend_excludes_seen_movies():
    model = SVDCollaborativeFiltering(n_factors=1, random_state=42)
    model.fit(_interactions())

    recs = model.recommend(1, _metadata(), top_n=2)

    assert not set(recs["movieID"]).intersection({10, 11})


def test_svd_fit_raises_on_missing_columns():
    model = SVDCollaborativeFiltering(n_factors=1, random_state=42)

    bad_df = pd.DataFrame(
        {
            "userID": [1, 2],
            "movieID": [10, 11],
        }
    )

    with pytest.raises(ValueError):
        model.fit(bad_df)


def test_sgd_fit_predict_and_recommend():
    model = SGDMatrixFactorization(n_factors=2, epochs=5, random_state=42)
    model.fit(_interactions())

    pred = model.predict_rating(1, 12)
    recs = model.recommend(1, _metadata(), top_n=1)

    assert isinstance(pred, float)
    assert model.global_mean > 0
    assert len(recs) == 1
    assert list(recs.columns) == ["movieID", "title", "predicted_rating"]


def test_sgd_predict_rating_returns_global_mean_for_unknown_ids():
    model = SGDMatrixFactorization(n_factors=2, epochs=5, random_state=42)
    model.fit(_interactions())

    assert model.predict_rating(999, 12) == pytest.approx(model.global_mean)
    assert model.predict_rating(1, 999) == pytest.approx(model.global_mean)


def test_sgd_recommend_excludes_seen_movies():
    model = SGDMatrixFactorization(n_factors=2, epochs=5, random_state=42)
    model.fit(_interactions())

    recs = model.recommend(1, _metadata(), top_n=2)

    assert not set(recs["movieID"]).intersection({10, 11})


def test_sgd_fit_raises_on_missing_columns():
    model = SGDMatrixFactorization(n_factors=2, epochs=5, random_state=42)

    bad_df = pd.DataFrame(
        {
            "userID": [1, 2],
            "movieID": [10, 11],
        }
    )

    with pytest.raises(ValueError):
        model.fit(bad_df)
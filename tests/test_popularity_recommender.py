import pandas as pd
import pytest

from src.models.popularity_recommender import PopularityRecommender


def _movie_features():
    return pd.DataFrame(
        {
            "movieID": [1, 2, 3, 4],
            "movie_avg_rating": [4.8, 4.2, 3.9, 4.6],
            "movie_rating_count": [100, 80, 60, 40],
        }
    )


def _metadata():
    return pd.DataFrame(
        {
            "movieID": [1, 2, 3, 4],
            "title": ["A", "B", "C", "D"],
        }
    )


def test_fit_returns_self_and_builds_recommendations():
    model = PopularityRecommender(min_votes=50, top_n=2)
    fitted = model.fit(_movie_features(), _metadata())

    assert fitted is model
    assert model.recommendations_df is not None
    assert "weighted_rating" in model.recommendations_df.columns
    assert len(model.recommendations_df) == 3


def test_recommend_returns_expected_columns_and_length():
    model = PopularityRecommender(min_votes=50, top_n=2)
    model.fit(_movie_features(), _metadata())

    recs = model.recommend()

    assert len(recs) == 2
    assert list(recs.columns) == [
        "movieID",
        "title",
        "movie_avg_rating",
        "movie_rating_count",
        "weighted_rating",
    ]


def test_recommend_orders_by_weighted_rating_descending():
    model = PopularityRecommender(min_votes=50, top_n=3)
    model.fit(_movie_features(), _metadata())

    recs = model.recommend()

    assert recs["weighted_rating"].is_monotonic_decreasing


def test_recommend_with_pool_size_samples_from_top_pool():
    model = PopularityRecommender(min_votes=50, top_n=2, pool_size=2)
    model.fit(_movie_features(), _metadata())

    recs = model.recommend(random_state=42)

    assert len(recs) == 2
    assert set(recs["movieID"]).issubset({1, 2})


def test_fit_raises_on_missing_required_columns():
    model = PopularityRecommender()

    movie_features = pd.DataFrame(
        {
            "movieID": [1, 2],
            "movie_avg_rating": [4.5, 4.0],
        }
    )
    metadata = _metadata()

    with pytest.raises(KeyError):
        model.fit(movie_features, metadata)


def test_recommend_raises_before_fit():
    model = PopularityRecommender()

    with pytest.raises(ValueError):
        model.recommend()
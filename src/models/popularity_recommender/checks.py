class PopularityRecommender:
    """
    Simple non-personalized recommendation system based on global popularity
    and IMDb-style weighted movie quality.

    This model recommends the same movies to every user and serves as a strong
    baseline before collaborative filtering models are introduced.

    Ranking is based on a weighted rating formula that balances:
    - average movie rating (quality)
    - number of ratings (popularity / confidence)

    This avoids unfairly promoting movies with very few but extremely high ratings.

    Parameters
    ----------
    min_votes : int, default=50
        Minimum number of ratings required for a movie to be considered
        reliable enough for recommendation.

    top_n : int, default=10
        Number of recommendations returned by default.
    pool_size : int, optional
        Size of the top-ranked pool to sample recommendations from. If None,
        recommendations will be sampled from the entire ranked list.
    """

    def __init__(self, min_votes=50, top_n=10, pool_size=None):
        self.min_votes = min_votes
        self.top_n = top_n
        self.pool_size = pool_size
        self.recommendations_df = None

    def fit(self, movie_features_df, movies_metadata_df):
        """
        Fit the popularity recommender by computing weighted movie rankings.

        Parameters
        ----------
        movie_features_df : pd.DataFrame
            Processed movie-level feature table containing:
            - movieID
            - movie_avg_rating
            - movie_rating_count

        movies_metadata_df : pd.DataFrame
            Raw movie metadata containing:
            - movieID
            - title

        Returns
        -------
        self
        """

        # Global mean rating across all movies (C)
        C = movie_features_df["movie_avg_rating"].mean()

        # Keep only sufficiently rated movies
        qualified = movie_features_df[
            movie_features_df["movie_rating_count"] >= self.min_votes
        ].copy()

        # IMDb-style weighted rating formula
        #
        # WR = (v / (v + m)) * R + (m / (v + m)) * C
        #
        # where:
        # WR = weighted rating
        # R  = movie average rating
        # v  = number of ratings
        # m  = minimum votes threshold
        # C  = global mean rating

        qualified["weighted_rating"] = (
            (
                qualified["movie_rating_count"]
                / (qualified["movie_rating_count"] + self.min_votes)
            ) * qualified["movie_avg_rating"]
            +
            (
                self.min_votes
                / (qualified["movie_rating_count"] + self.min_votes)
            ) * C
        )

        # Merge titles for readability
        self.recommendations_df = (
            qualified.merge(
                movies_metadata_df,
                on="movieID",
                how="left"
            )
            .sort_values(
                by="weighted_rating",
                ascending=False
            )
            .reset_index(drop=True)
        )

        return self

    def recommend(self, top_n=None, random_state=None):
        """
        Generate movie recommendations.

        If `self.pool_size` is None, returns the top-N globally highest-rated movies.
        If `self.pool_size` is set, randomly samples `top_n` movies from the top `self.pool_size` movies,
        introducing diversity in the recommendations.

        Parameters
        ----------
        top_n : int, optional
            Number of recommendations to return. Defaults to `self.top_n`.

        random_state : int, optional
            Random seed for reproducibility when sampling from the pool.

        Returns
        -------
        pd.DataFrame
            DataFrame containing the recommended movies with columns:
            'movieID', 'title', 'movie_avg_rating', 'movie_rating_count', 'weighted_rating'.
        """

        if self.recommendations_df is None:
            raise ValueError("Model must be fitted before calling recommend().")

        if top_n is None:
            top_n = self.top_n

        # If no pool is defined, return the top-N globally best movies
        if self.pool_size is None:
            return self.recommendations_df.head(top_n)[
                [
                    "movieID",
                    "title",
                    "movie_avg_rating",
                    "movie_rating_count",
                    "weighted_rating"
                ]
            ].reset_index(drop=True)

        # Otherwise, sample top_n movies from the top `self.pool_size` movies
        pool = self.recommendations_df.head(self.pool_size)

        if len(pool) <= top_n:
            result = pool.head(top_n)
        else:
            result = pool.sample(n=top_n, random_state=random_state)

        return result[
            [
                "movieID",
                "title",
                "movie_avg_rating",
                "movie_rating_count",
                "weighted_rating"
        ]
    ].reset_index(drop=True)
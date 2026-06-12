import mlflow


class PopularityRecommender:
    """
    Simple non-personalized recommendation system based on global popularity
    and IMDb-style weighted movie quality.

    MLflow tracking is built-in:
    - Parameters: min_votes, top_n, pool_size
    - Metrics:    n_qualified_movies, mean_weighted_rating, top1_weighted_rating
    - Tags:       model_type = "PopularityRecommender"

    Usage
    -----
    model = PopularityRecommender(min_votes=50, top_n=10)
    model.fit(movie_features_df, movies_metadata_df)   # auto-logged to MLflow
    model.recommend()
    """

    def __init__(self, min_votes=50, top_n=10, pool_size=None):
        self.min_votes = min_votes
        self.top_n = top_n
        self.pool_size = pool_size
        self.recommendations_df = None

    def fit(self, movie_features_df, movies_metadata_df):
        """
        Fit the popularity recommender and log parameters/metrics to MLflow.

        Parameters
        ----------
        movie_features_df : pd.DataFrame
            Must contain: movieID, movie_avg_rating, movie_rating_count

        movies_metadata_df : pd.DataFrame
            Must contain: movieID, title

        Returns
        -------
        self
        """

        mlflow.start_run(run_name="PopularityRecommender")

        try:
            mlflow.log_params(
                {
                    "model_type": "PopularityRecommender",
                    "min_votes": self.min_votes,
                    "top_n": self.top_n,
                    "pool_size": self.pool_size if self.pool_size is not None else "None",
                }
            )

            mlflow.set_tag("model_type", "PopularityRecommender")

            C = movie_features_df["movie_avg_rating"].mean()
            mlflow.log_metric("global_mean_rating", round(C, 4))

            qualified = movie_features_df[movie_features_df["movie_rating_count"] >= self.min_votes].copy()

            n_total = len(movie_features_df)
            n_qualified = len(qualified)
            print(f"Movies passing min_votes={self.min_votes} threshold: {n_qualified} / {n_total}")

            mlflow.log_metrics(
                {
                    "n_total_movies": n_total,
                    "n_qualified_movies": n_qualified,
                }
            )

            qualified["weighted_rating"] = (
                qualified["movie_rating_count"] / (qualified["movie_rating_count"] + self.min_votes)
            ) * qualified["movie_avg_rating"] + (
                self.min_votes / (qualified["movie_rating_count"] + self.min_votes)
            ) * C

            self.recommendations_df = (
                qualified.merge(movies_metadata_df, on="movieID", how="left")
                .sort_values(by="weighted_rating", ascending=False)
                .reset_index(drop=True)
            )

            mlflow.log_metrics(
                {
                    "mean_weighted_rating": round(self.recommendations_df["weighted_rating"].mean(), 4),
                    "top1_weighted_rating": round(self.recommendations_df["weighted_rating"].iloc[0], 4),
                }
            )

            return self
        finally:
            mlflow.end_run()

    def recommend(self, top_n=None, random_state=None):
        """
        Generate movie recommendations.

        If pool_size is None, returns the globally top-ranked movies.
        If pool_size is set, randomly samples top_n from the top pool_size movies.

        Parameters
        ----------
        top_n : int, optional
        random_state : int, optional

        Returns
        -------
        pd.DataFrame with columns:
            movieID, title, movie_avg_rating, movie_rating_count, weighted_rating
        """

        if self.recommendations_df is None:
            raise ValueError("Model must be fitted before calling recommend().")

        if top_n is None:
            top_n = self.top_n

        cols = ["movieID", "title", "movie_avg_rating", "movie_rating_count", "weighted_rating"]

        if self.pool_size is None:
            return self.recommendations_df.head(top_n)[cols].reset_index(drop=True)

        pool = self.recommendations_df.head(self.pool_size)

        if len(pool) <= top_n:
            result = pool.head(top_n)
        else:
            result = pool.sample(n=top_n, random_state=random_state)

        return result[cols].reset_index(drop=True)

    def recommend_one(self, pool_size=100, random_state=None):
        """
        Recommend a single movie sampled from the top pool.

        Parameters
        ----------
        pool_size : int, default=100
            Number of top-ranked movies considered.

        random_state : int, optional

        Returns
        -------
        pd.Series
            One recommended movie.
        """

        if self.recommendations_df is None:
            raise ValueError("Model must be fitted before calling recommend_one().")

        cols = ["movieID", "title", "movie_avg_rating", "movie_rating_count", "weighted_rating"]

        pool = self.recommendations_df.head(pool_size)

        return pool.sample(n=1, random_state=random_state)[cols].iloc[0]

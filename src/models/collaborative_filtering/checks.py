from typing import Any, Optional

import numpy as np
import pandas as pd

from sklearn.metrics import mean_absolute_error, mean_squared_error

from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds


class SVDCollaborativeFiltering:
    """
    Collaborative Filtering recommender using Matrix Factorization
    via truncated Singular Value Decomposition (SVD).

    This model learns latent user and item representations from the
    user-item rating matrix and predicts ratings for unseen movies.

    Workflow
    --------
    1. Create user-item matrix
       rows    -> users
       columns -> movies
       values  -> ratings

    2. Mean-center user ratings
       This removes personal rating bias:
       some users rate everything high,
       others rate everything low.

    3. Convert matrix to sparse representation

    4. Apply truncated SVD:

       R ≈ U Σ Vᵀ

    5. Reconstruct dense prediction matrix:

       R̂ = U Σ Vᵀ + user_mean

    6. Recommend highest predicted unseen items

    Parameters
    ----------
    n_factors : int, default=50
        Number of latent factors (k).

        Higher values:
        - better reconstruction
        - higher overfitting risk

        Lower values:
        - stronger generalization
        - possible underfitting

    random_state : int, default=42
        Reproducibility seed.
    """

    def __init__(
        self,
        n_factors=50,
        random_state=42
    ):
        self.n_factors = n_factors
        self.random_state = random_state

        self.user_item_matrix = None
        self.user_means = None
        self.predictions_df = None
        self.train_df = None

    def fit(self, interactions_df):
        """
        Fit collaborative filtering model.

        Parameters
        ----------
        interactions_df : pd.DataFrame

        Required columns:
        - userID
        - movieID
        - interaction_rating

        Returns
        -------
        self
        """

        print("Step 1: Building user-item matrix...")

        required_cols = [
            "userID",
            "movieID",
            "interaction_rating"
        ]

        for col in required_cols:
            if col not in interactions_df.columns:
                raise ValueError(
                    f"Missing required column: {col}"
                )

        self.train_df = interactions_df.copy()

        # Create pivot table

        self.user_item_matrix = interactions_df.pivot_table(
            index="userID",
            columns="movieID",
            values="interaction_rating"
        )

        print(
            f"User-item matrix shape: "
            f"{self.user_item_matrix.shape}"
        )

        # Mean-centering

        print("Step 2: Mean-centering user ratings...")

        self.user_means = self.user_item_matrix.mean(axis=1)

        matrix_demeaned = (
            self.user_item_matrix
            .sub(self.user_means, axis=0)
            .fillna(0)
        )

        # Sparse matrix conversion

        print("Step 3: Converting to sparse matrix...")

        sparse_matrix = csr_matrix(
            matrix_demeaned.values
        )

        # Truncated SVD

        print(
            f"Step 4: Running truncated SVD "
            f"with k={self.n_factors}..."
        )

        U, sigma, Vt = svds(
            sparse_matrix,
            k=self.n_factors
        )

        sigma = np.diag(sigma)

        print("SVD factorization completed.")

        # Matrix reconstruction

        print("Step 5: Reconstructing predictions...")

        reconstructed = (
            np.dot(
                np.dot(U, sigma),
                Vt
            )
            + self.user_means.values.reshape(-1, 1)
        )

        self.predictions_df = pd.DataFrame(
            reconstructed,
            index=self.user_item_matrix.index,
            columns=self.user_item_matrix.columns
        )

        print("Prediction matrix created successfully.")

        return self

    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a single user-movie pair.

        Predictions are post-processed to:

        1. clip values to valid range
        [0.5, 5.0]

        2. round to nearest 0.5 increment

        Parameters
        ----------
        user_id : int

        movie_id : int

        Returns
        -------
        float
        """

        if self.predictions_df is None:
            raise ValueError(
                "Model must be fitted first."
            )

        if user_id not in self.predictions_df.index:
            return np.nan

        if movie_id not in self.predictions_df.columns:
            return np.nan

        prediction = self.predictions_df.loc[
            user_id,
            movie_id
        ]

        prediction = np.clip(
            prediction,
            0.5,
            5.0
        )

        prediction = round(prediction * 2) / 2

        return float(prediction)

    def evaluate(self, test_df):
        """
        Evaluate model using RMSE and MAE.

        Parameters
        ----------
        test_df : pd.DataFrame

        Returns
        -------
        dict
        """

        print("Evaluating model on test set...")

        predictions = []
        actuals = []

        for row in test_df.itertuples():
            pred = self.predict_rating(
                row.userID,
                row.movieID
            )

            if not np.isnan(pred):
                predictions.append(pred)
                actuals.append(row.interaction_rating)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(
            np.mean((actuals - predictions) ** 2)
        )

        mae = np.mean(
            np.abs(actuals - predictions)
        )

        results = {
            "RMSE": rmse,
            "MAE": mae,
            "Evaluated_Interactions": len(actuals)
        }

        print(results)

        return results

    def recommend(
        self,
        user_id,
        movies_metadata_df,
        top_n=10
    ):
        """
        Generate Top-N personalized recommendations.

        Parameters
        ----------
        user_id : int

        movies_metadata_df : pd.DataFrame
            Must contain:
            - movieID
            - title

        top_n : int

        Returns
        -------
        pd.DataFrame
        """

        if self.predictions_df is None:
            raise ValueError(
                "Model must be fitted first."
            )

        if user_id not in self.predictions_df.index:
            raise ValueError(
                f"User {user_id} not found."
            )

        rated_movies = set(
            self.train_df[
                self.train_df["userID"] == user_id
            ]["movieID"].unique()
        )

        user_predictions = (
            self.predictions_df
            .loc[user_id]
            .drop(
                labels=rated_movies,
                errors="ignore"
            )
            .sort_values(ascending=False)
            .reset_index()
        )

        user_predictions.columns = [
            "movieID",
            "predicted_rating"
        ]

        recommendations = (
            user_predictions
            .merge(
                movies_metadata_df,
                on="movieID",
                how="left"
            )
            .head(top_n)
        )

        return recommendations[
            [
                "movieID",
                "title",
                "predicted_rating"
            ]
        ]
    




class SGDMatrixFactorization:
    """
    Collaborative Filtering recommender using Matrix Factorization
    optimized with Stochastic Gradient Descent (SGD),
    commonly known as Funk SVD.

    This model learns latent user and item representations directly
    from observed ratings without reconstructing the full dense
    user-item matrix.

    Compared to classical truncated SVD, this approach is more suitable
    for recommendation systems because it:

    - trains only on known interactions
    - handles sparse data efficiently
    - includes user and item bias correction
    - supports regularization against overfitting
    - scales better for large datasets

    ------------------------------------------------------------------
    Prediction Formula
    ------------------------------------------------------------------

    Predicted rating for user u and movie i:

        r̂_ui = μ + b_u + b_i + (P_u · Q_i)

    where:

    μ   = global mean rating
    b_u = user bias
    b_i = item bias
    P_u = latent preference vector of user u
    Q_i = latent attribute vector of movie i

    Interpretation:

    - some users consistently rate higher/lower than others
    - some movies are generally better received
    - latent vectors capture hidden preference patterns

    The dot product:

        P_u · Q_i

    models the compatibility between user preferences and
    movie characteristics in latent space.

    ------------------------------------------------------------------
    Parameters
    ------------------------------------------------------------------

    n_factors : int, default=50
        Number of latent factors (embedding dimensions).

        Higher values:
        - more expressive model
        - higher overfitting risk

        Lower values:
        - stronger generalization
        - possible underfitting

    learning_rate : float, default=0.005
        SGD learning rate.

    regularization : float, default=0.02
        L2 regularization strength.

    epochs : int, default=20
        Number of full passes through training data.

    rating_min : float, default=0.5
        Minimum valid rating value.

    rating_max : float, default=5.0
        Maximum valid rating value.

    random_state : int, default=42
        Reproducibility seed.

    ------------------------------------------------------------------
    Attributes
    ------------------------------------------------------------------

    global_mean : float
        Global average rating.

    user_factors : np.ndarray
        User latent matrix P.

    movie_factors : np.ndarray
        Item latent matrix Q.

    user_biases : np.ndarray
        User bias vector.

    movie_biases : np.ndarray
        Movie bias vector.

    train_df : pd.DataFrame
        Stored training interactions for recommendation filtering.
    """

    def __init__(
        self,
        n_factors: int = 50,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        epochs: int = 20,
        rating_min: float = 0.5,
        rating_max: float = 5.0,
        random_state: int = 42
    ):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.epochs = epochs
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.random_state = random_state

        # ID mappings
        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}

        # Model parameters
        self.global_mean = 0.0
        self.user_factors = None
        self.movie_factors = None
        self.user_biases = None
        self.movie_biases = None

        # Stored training data
        self.train_df = None

    def fit(self, interactions_df: pd.DataFrame):
        """
        Train matrix factorization model using SGD.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Must contain:

            - userID
            - movieID
            - interaction_rating

        Returns
        -------
        self
        """

        print("Step 1: Preparing training data...")

        required_cols = [
            "userID",
            "movieID",
            "interaction_rating"
        ]

        for col in required_cols:
            if col not in interactions_df.columns:
                raise ValueError(f"Missing required column: {col}")

        self.train_df = interactions_df.copy()

        # Create user/movie index mappings

        unique_users = interactions_df["userID"].unique()
        unique_movies = interactions_df["movieID"].unique()

        self.user_to_idx = {
            user: idx
            for idx, user in enumerate(unique_users)
        }

        self.movie_to_idx = {
            movie: idx
            for idx, movie in enumerate(unique_movies)
        }

        self.idx_to_movie = {
            idx: movie
            for movie, idx in self.movie_to_idx.items()
        }

        n_users = len(unique_users)
        n_movies = len(unique_movies)

        print(f"Users: {n_users}")
        print(f"Movies: {n_movies}")

        # Initialize parameters

        print("Step 2: Initializing latent factors...")

        self.global_mean = interactions_df[
            "interaction_rating"
        ].mean()

        rng = np.random.RandomState(self.random_state)

        init_scale = np.sqrt(1 / self.n_factors)

        self.user_factors = rng.normal(
            scale=init_scale,
            size=(n_users, self.n_factors)
        )

        self.movie_factors = rng.normal(
            scale=init_scale,
            size=(n_movies, self.n_factors)
        )

        self.user_biases = np.zeros(n_users)
        self.movie_biases = np.zeros(n_movies)

        # Prepare samples for SGD

        user_idx = interactions_df["userID"].map(
            self.user_to_idx
        ).values

        movie_idx = interactions_df["movieID"].map(
            self.movie_to_idx
        ).values

        ratings = interactions_df[
            "interaction_rating"
        ].values

        samples = list(
            zip(user_idx, movie_idx, ratings)
        )

        # Training loop

        print(
            f"Step 3: Training model "
            f"for {self.epochs} epochs..."
        )

        for epoch in range(self.epochs):
            np.random.shuffle(samples)

            squared_error = 0.0

            for u, i, r_true in samples:

                # Prediction
                dot_product = np.dot(
                    self.user_factors[u],
                    self.movie_factors[i]
                )

                r_pred = (
                    self.global_mean
                    + self.user_biases[u]
                    + self.movie_biases[i]
                    + dot_product
                )

                # Error
                error = r_true - r_pred
                squared_error += error ** 2

                # Update biases
                self.user_biases[u] += self.lr * (
                    error
                    - self.reg * self.user_biases[u]
                )

                self.movie_biases[i] += self.lr * (
                    error
                    - self.reg * self.movie_biases[i]
                )

                # Update latent factors
                old_user_vector = self.user_factors[u].copy()

                self.user_factors[u] += self.lr * (
                    error * self.movie_factors[i]
                    - self.reg * self.user_factors[u]
                )

                self.movie_factors[i] += self.lr * (
                    error * old_user_vector
                    - self.reg * self.movie_factors[i]
                )

            rmse = np.sqrt(
                squared_error / len(samples)
            )

            if epoch == 0 or (epoch + 1) % 5 == 0:
                print(
                    f"Epoch {epoch + 1:02d}/{self.epochs} "
                    f"| Training RMSE: {rmse:.4f}"
                )

        print("Training completed successfully.")

        return self

    def predict_rating(
        self,
        user_id: Any,
        movie_id: Any
    ) -> float:
        """
        Predict rating for a single user-movie pair.

        Cold-start strategy:
        if user or movie is unseen,
        fallback to global mean rating.

        Parameters
        ----------
        user_id : Any

        movie_id : Any

        Returns
        -------
        float
        """

        if (
            user_id not in self.user_to_idx
            or movie_id not in self.movie_to_idx
        ):
            return self.global_mean

        u = self.user_to_idx[user_id]
        i = self.movie_to_idx[movie_id]

        prediction = (
            self.global_mean
            + self.user_biases[u]
            + self.movie_biases[i]
            + np.dot(
                self.user_factors[u],
                self.movie_factors[i]
            )
        )

        prediction = np.clip(
            prediction,
            self.rating_min,
            self.rating_max
        )

        return float(prediction)

    def evaluate(
        self,
        test_df: pd.DataFrame
    ) -> dict:
        """
        Evaluate model performance on test set.

        Metrics:

        - RMSE
        - MAE

        Parameters
        ----------
        test_df : pd.DataFrame

        Returns
        -------
        dict
        """

        print("Evaluating model on test set...")

        y_true = []
        y_pred = []

        for row in test_df.itertuples():
            pred = self.predict_rating(
                user_id=row.userID,
                movie_id=row.movieID
            )

            y_true.append(row.interaction_rating)
            y_pred.append(pred)

        rmse = np.sqrt(
            mean_squared_error(y_true, y_pred)
        )

        mae = mean_absolute_error(
            y_true,
            y_pred
        )

        results = {
            "RMSE": round(rmse, 4),
            "MAE": round(mae, 4)
        }

        print(results)

        return results

    def recommend(
        self,
        user_id: Any,
        movies_metadata_df: pd.DataFrame,
        top_n: int = 10
    ) -> pd.DataFrame:
        """
        Generate Top-N personalized recommendations.

        Already rated movies are automatically excluded.

        Parameters
        ----------
        user_id : Any

        movies_metadata_df : pd.DataFrame
            Must contain:

            - movieID
            - title

        top_n : int, default=10

        Returns
        -------
        pd.DataFrame
        """

        if user_id not in self.user_to_idx:
            raise ValueError(
                f"User {user_id} not found."
            )

        seen_movies = set(
            self.train_df[
                self.train_df["userID"] == user_id
            ]["movieID"].unique()
        )

        candidate_movies = [
            movie_id
            for movie_id in self.movie_to_idx.keys()
            if movie_id not in seen_movies
        ]

        predictions = []

        for movie_id in candidate_movies:
            pred = self.predict_rating(
                user_id,
                movie_id
            )

            predictions.append(
                (movie_id, pred)
            )

        recommendations = pd.DataFrame(
            predictions,
            columns=[
                "movieID",
                "predicted_rating"
            ]
        )

        recommendations = (
            recommendations
            .merge(
                movies_metadata_df,
                on="movieID",
                how="left"
            )
            .sort_values(
                by="predicted_rating",
                ascending=False
            )
            .head(top_n)
            .reset_index(drop=True)
        )

        return recommendations[
            [
                "movieID",
                "title",
                "predicted_rating"
            ]
        ]

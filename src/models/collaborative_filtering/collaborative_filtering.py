from typing import Any

import mlflow
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds
from sklearn.metrics import mean_absolute_error, mean_squared_error


class SVDCollaborativeFiltering:
    """
    Collaborative Filtering recommender using Matrix Factorization
    via truncated Singular Value Decomposition (SVD).

    MLflow tracking is built-in:
    - Parameters: n_factors, random_state, n_users, n_movies, n_interactions
    - Metrics:    test_rmse, test_mae, evaluated_interactions (logged in evaluate())
    - Tags:       model_type = "SVD"

    Usage
    -----
    model = SVDCollaborativeFiltering(n_factors=50)
    model.fit(train_df)
    model.evaluate(test_df)          # metrics auto-logged to MLflow
    model.recommend(user_id, meta_df)
    """

    def __init__(self, n_factors=50, random_state=42):
        self.n_factors = n_factors
        self.random_state = random_state

        self.user_item_matrix = None
        self.user_means = None
        self.predictions_df = None
        self.train_df = None

    def fit(self, interactions_df):
        """
        Fit collaborative filtering model and start an MLflow run.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Required columns: userID, movieID, interaction_rating

        Returns
        -------
        self
        """

        # Start MLflow run — all subsequent log_* calls go into this run
        mlflow.start_run(run_name="SVD_CollaborativeFiltering")

        try:
            # Log hyperparameters
            mlflow.log_params({
                "model_type":   "SVD",
                "n_factors":    self.n_factors,
                "random_state": self.random_state,
            })

            mlflow.set_tag("model_type", "SVD")

            print("Step 1: Building user-item matrix...")

            required_cols = ["userID", "movieID", "interaction_rating"]
            for col in required_cols:
                if col not in interactions_df.columns:
                    raise ValueError(f"Missing required column: {col}")

            self.train_df = interactions_df.copy()

            self.user_item_matrix = interactions_df.pivot_table(
                index="userID", columns="movieID", values="interaction_rating"
            )

            n_users, n_movies = self.user_item_matrix.shape
            print(f"User-item matrix shape: {self.user_item_matrix.shape}")

            # Log dataset statistics
            mlflow.log_params({
                "n_users":        n_users,
                "n_movies":       n_movies,
                "n_interactions": len(interactions_df),
            })

            print("Step 2: Mean-centering user ratings...")
            self.user_means = self.user_item_matrix.mean(axis=1)
            matrix_demeaned = self.user_item_matrix.sub(self.user_means, axis=0).fillna(0)

            print("Step 3: Converting to sparse matrix...")
            sparse_matrix = csr_matrix(matrix_demeaned.values)

            print(f"Step 4: Running truncated SVD with k={self.n_factors}...")
            U, sigma, Vt = svds(sparse_matrix, k=self.n_factors)
            sigma = np.diag(sigma)
            print("SVD factorization completed.")

            print("Step 5: Reconstructing predictions...")
            reconstructed = np.dot(np.dot(U, sigma), Vt) + self.user_means.values.reshape(-1, 1)

            self.predictions_df = pd.DataFrame(
                reconstructed,
                index=self.user_item_matrix.index,
                columns=self.user_item_matrix.columns,
            )

            print("Prediction matrix created successfully.")

        except Exception:
            # Make sure the run is closed even if something goes wrong
            mlflow.end_run(status="FAILED")
            raise

        return self

    def predict_rating(self, user_id, movie_id):
        """
        Predict rating for a single user-movie pair.
        Clips to [0.5, 5.0] and rounds to nearest 0.5.
        """

        if self.predictions_df is None:
            raise ValueError("Model must be fitted first.")

        if user_id not in self.predictions_df.index:
            return np.nan
        if movie_id not in self.predictions_df.columns:
            return np.nan

        prediction = self.predictions_df.loc[user_id, movie_id]
        prediction = np.clip(prediction, 0.5, 5.0)
        prediction = round(prediction * 2) / 2

        return float(prediction)

    def evaluate(self, test_df):
        """
        Evaluate model using RMSE and MAE, and log metrics to MLflow.

        Parameters
        ----------
        test_df : pd.DataFrame

        Returns
        -------
        dict with keys: RMSE, MAE, Evaluated_Interactions
        """

        print("Evaluating model on test set...")

        predictions = []
        actuals = []

        for row in test_df.itertuples():
            pred = self.predict_rating(row.userID, row.movieID)
            if not np.isnan(pred):
                predictions.append(pred)
                actuals.append(row.interaction_rating)

        predictions = np.array(predictions)
        actuals = np.array(actuals)

        rmse = np.sqrt(np.mean((actuals - predictions) ** 2))
        mae = np.mean(np.abs(actuals - predictions))

        results = {
            "RMSE":                   rmse,
            "MAE":                    mae,
            "Evaluated_Interactions": len(actuals),
        }

        # Log evaluation metrics to MLflow
        mlflow.log_metrics({
            "test_rmse":              round(rmse, 4),
            "test_mae":               round(mae, 4),
            "evaluated_interactions": len(actuals),
        })

        print(results)

        # End the MLflow run after evaluation
        mlflow.end_run()

        return results

    def recommend(self, user_id, movies_metadata_df, top_n=10):
        """
        Generate Top-N personalized recommendations.

        Parameters
        ----------
        user_id : int
        movies_metadata_df : pd.DataFrame  (must contain: movieID, title)
        top_n : int

        Returns
        -------
        pd.DataFrame with columns: movieID, title, predicted_rating
        """

        if self.predictions_df is None:
            raise ValueError("Model must be fitted first.")
        if user_id not in self.predictions_df.index:
            raise ValueError(f"User {user_id} not found.")

        rated_movies = set(self.train_df[self.train_df["userID"] == user_id]["movieID"].unique())

        user_predictions = (
            self.predictions_df.loc[user_id]
            .drop(labels=rated_movies, errors="ignore")
            .sort_values(ascending=False)
            .reset_index()
        )
        user_predictions.columns = ["movieID", "predicted_rating"]

        recommendations = user_predictions.merge(movies_metadata_df, on="movieID", how="left").head(top_n)

        return recommendations[["movieID", "title", "predicted_rating"]]


class SGDMatrixFactorization:
    """
    Collaborative Filtering recommender using Matrix Factorization
    optimized with Stochastic Gradient Descent (SGD) — Funk SVD.

    MLflow tracking is built-in:
    - Parameters: n_factors, learning_rate, regularization, epochs,
                  rating_min, rating_max, random_state
    - Metrics:    train_rmse logged per epoch, test_rmse and test_mae
                  logged after evaluate()
    - Tags:       model_type = "SGD"

    Prediction Formula
    ------------------
        r̂_ui = μ + b_u + b_i + (P_u · Q_i)

    Usage
    -----
    model = SGDMatrixFactorization(n_factors=50, epochs=20)
    model.fit(train_df)
    model.evaluate(test_df)          # metrics auto-logged to MLflow
    model.recommend(user_id, meta_df)
    """

    def __init__(
        self,
        n_factors: int = 50,
        learning_rate: float = 0.005,
        regularization: float = 0.02,
        epochs: int = 20,
        rating_min: float = 0.5,
        rating_max: float = 5.0,
        random_state: int = 42,
    ):
        self.n_factors = n_factors
        self.lr = learning_rate
        self.reg = regularization
        self.epochs = epochs
        self.rating_min = rating_min
        self.rating_max = rating_max
        self.random_state = random_state

        self.user_to_idx = {}
        self.movie_to_idx = {}
        self.idx_to_movie = {}

        self.global_mean = 0.0
        self.user_factors = None
        self.movie_factors = None
        self.user_biases = None
        self.movie_biases = None

        self.train_df = None

    def fit(self, interactions_df: pd.DataFrame):
        """
        Train matrix factorization model using SGD, with per-epoch MLflow logging.

        Parameters
        ----------
        interactions_df : pd.DataFrame
            Must contain: userID, movieID, interaction_rating

        Returns
        -------
        self
        """

        # Start MLflow run
        mlflow.start_run(run_name="SGD_MatrixFactorization")

        try:
            # Log all hyperparameters
            mlflow.log_params({
                "model_type":     "SGD",
                "n_factors":      self.n_factors,
                "learning_rate":  self.lr,
                "regularization": self.reg,
                "epochs":         self.epochs,
                "rating_min":     self.rating_min,
                "rating_max":     self.rating_max,
                "random_state":   self.random_state,
            })

            mlflow.set_tag("model_type", "SGD")

            print("Step 1: Preparing training data...")

            required_cols = ["userID", "movieID", "interaction_rating"]
            for col in required_cols:
                if col not in interactions_df.columns:
                    raise ValueError(f"Missing required column: {col}")

            self.train_df = interactions_df.copy()

            unique_users = interactions_df["userID"].unique()
            unique_movies = interactions_df["movieID"].unique()

            self.user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
            self.movie_to_idx = {movie: idx for idx, movie in enumerate(unique_movies)}
            self.idx_to_movie = {idx: movie for movie, idx in self.movie_to_idx.items()}

            n_users = len(unique_users)
            n_movies = len(unique_movies)

            print(f"Users: {n_users}")
            print(f"Movies: {n_movies}")

            # Log dataset statistics
            mlflow.log_params({
                "n_users":        n_users,
                "n_movies":       n_movies,
                "n_interactions": len(interactions_df),
            })

            print("Step 2: Initializing latent factors...")

            self.global_mean = interactions_df["interaction_rating"].mean()
            mlflow.log_metric("global_mean_rating", round(self.global_mean, 4))

            rng = np.random.RandomState(self.random_state)
            init_scale = np.sqrt(1 / self.n_factors)

            self.user_factors = rng.normal(scale=init_scale, size=(n_users, self.n_factors))
            self.movie_factors = rng.normal(scale=init_scale, size=(n_movies, self.n_factors))
            self.user_biases = np.zeros(n_users)
            self.movie_biases = np.zeros(n_movies)

            user_idx = interactions_df["userID"].map(self.user_to_idx).values
            movie_idx = interactions_df["movieID"].map(self.movie_to_idx).values
            ratings = interactions_df["interaction_rating"].values
            samples = list(zip(user_idx, movie_idx, ratings))

            print(f"Step 3: Training model for {self.epochs} epochs...")

            for epoch in range(self.epochs):
                np.random.shuffle(samples)
                squared_error = 0.0

                for u, i, r_true in samples:
                    dot_product = np.dot(self.user_factors[u], self.movie_factors[i])
                    r_pred = self.global_mean + self.user_biases[u] + self.movie_biases[i] + dot_product

                    error = r_true - r_pred
                    squared_error += error ** 2

                    self.user_biases[u] += self.lr * (error - self.reg * self.user_biases[u])
                    self.movie_biases[i] += self.lr * (error - self.reg * self.movie_biases[i])

                    old_user_vector = self.user_factors[u].copy()
                    self.user_factors[u] += self.lr * (error * self.movie_factors[i] - self.reg * self.user_factors[u])
                    self.movie_factors[i] += self.lr * (error * old_user_vector - self.reg * self.movie_factors[i])

                rmse = np.sqrt(squared_error / len(samples))

                # Log training RMSE every epoch — shows as a curve in MLflow UI
                mlflow.log_metric("train_rmse", round(rmse, 4), step=epoch + 1)

                if epoch == 0 or (epoch + 1) % 5 == 0:
                    print(f"Epoch {epoch + 1:02d}/{self.epochs} | Training RMSE: {rmse:.4f}")

            print("Training completed successfully.")

        except Exception:
            # Make sure the run is closed even if something goes wrong
            mlflow.end_run(status="FAILED")
            raise

        return self

    def predict_rating(self, user_id: Any, movie_id: Any) -> float:
        """
        Predict rating for a single user-movie pair.
        Falls back to global mean for unseen users/movies.
        """

        if user_id not in self.user_to_idx or movie_id not in self.movie_to_idx:
            return self.global_mean

        u = self.user_to_idx[user_id]
        i = self.movie_to_idx[movie_id]

        prediction = (
            self.global_mean
            + self.user_biases[u]
            + self.movie_biases[i]
            + np.dot(self.user_factors[u], self.movie_factors[i])
        )

        prediction = np.clip(prediction, self.rating_min, self.rating_max)

        return float(prediction)

    def evaluate(self, test_df: pd.DataFrame) -> dict:
        """
        Evaluate model performance on test set and log metrics to MLflow.

        Parameters
        ----------
        test_df : pd.DataFrame

        Returns
        -------
        dict with keys: RMSE, MAE
        """

        print("Evaluating model on test set...")

        y_true = []
        y_pred = []

        for row in test_df.itertuples():
            pred = self.predict_rating(user_id=row.userID, movie_id=row.movieID)
            y_true.append(row.interaction_rating)
            y_pred.append(pred)

        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        results = {"RMSE": round(rmse, 4), "MAE": round(mae, 4)}

        # Log test metrics to MLflow
        mlflow.log_metrics({
            "test_rmse": round(rmse, 4),
            "test_mae":  round(mae, 4),
        })

        print(results)

        # End the MLflow run after evaluation
        mlflow.end_run()

        return results

    def recommend(self, user_id: Any, movies_metadata_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """
        Generate Top-N personalized recommendations (already rated movies excluded).

        Parameters
        ----------
        user_id : Any
        movies_metadata_df : pd.DataFrame  (must contain: movieID, title)
        top_n : int, default=10

        Returns
        -------
        pd.DataFrame with columns: movieID, title, predicted_rating
        """

        if user_id not in self.user_to_idx:
            raise ValueError(f"User {user_id} not found.")

        seen_movies = set(self.train_df[self.train_df["userID"] == user_id]["movieID"].unique())
        candidate_movies = [m for m in self.movie_to_idx.keys() if m not in seen_movies]

        predictions = [(m, self.predict_rating(user_id, m)) for m in candidate_movies]

        recommendations = pd.DataFrame(predictions, columns=["movieID", "predicted_rating"])
        recommendations = (
            recommendations.merge(movies_metadata_df, on="movieID", how="left")
            .sort_values(by="predicted_rating", ascending=False)
            .head(top_n)
            .reset_index(drop=True)
        )

        return recommendations[["movieID", "title", "predicted_rating"]]

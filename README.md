# Explainable Recsys

Explainable Recsys is a movie recommendation system project built around the HetRec-2011 MovieLens dataset. The repository combines exploratory data analysis, feature engineering, multiple recommendation approaches, and explainability methods to make recommendations more interpretable and easier to compare.

## License

This project is licensed under the MIT License.

## Authors

- Farah Abid
- Mateusz Skrzątek
- Michał Raczkiewicz
- Krzysztof Bachanek

## Overview

The project includes:

- exploratory data analysis for the raw dataset tables
- feature engineering for users, movies, and interactions
- a popularity-based baseline recommender
- collaborative filtering models
- content-based recommendation
- explanation-oriented tooling for model interpretation
- MLflow tracking for experiment logging and comparison
- unit tests for the main recommendation models

## Requirements

- Python 3.11+
- `pip`
- optional: MLflow for experiment inspection

## Installation

### 1. Create a virtual environment

```bash
python -m venv .venv
```

### 2. Activate the environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Running Tests

```bash
pytest
```

### Linting

```bash
ruff check .
```

### Formatting

```bash
ruff format .
```

## Project Structure

```text
.
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── scripts/
├── src/
│   ├── content_based/
│   ├── eda/
│   ├── features/
│   └── models/
├── tests/
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Main Modules

### `src/eda/`

Tools for:

- loading datasets
- data quality checks
- validation of source tables
- preparing data for downstream processing

### `src/features/`

Feature engineering for:

- users
- movies
- interactions

The goal is to create features that are:

- useful for models
- readable
- reusable across experiments

### `src/content_based/`

Content-based recommendation utilities and supporting code for:

- feature representation
- embeddings
- evaluation
- explainability

### `src/models/`

Recommendation models:

- `PopularityRecommender`
- `SVDCollaborativeFiltering`
- `SGDMatrixFactorization`

The models log parameters and metrics to MLflow.

## Models

### Popularity Recommender

A non-personalized baseline based on popularity and weighted rating quality.

**Use cases:**

- quick baseline
- comparison against more complex models

### Collaborative Filtering

Two matrix factorization variants:

- `SVDCollaborativeFiltering`
- `SGDMatrixFactorization`

**Use cases:**

- personalized recommendations
- prediction quality comparison
- trade-off analysis between accuracy and interpretability

### Content-Based Recommendation

A feature-based approach that uses movie content and derived representations.

**Use cases:**

- similarity-based recommendations
- explainability support
- comparison with collaborative filtering models

## Explainability

The project is designed to support interpretation at the feature level.

Supported elements include:

- human-readable input features
- feature importance analysis
- SHAP-based explanations
- LIME-based explanations
- traceable experiment logging

## Dataset

The project is based on the HetRec-2011 MovieLens dataset.

Used tables:

- movies
- movie_actors
- movie_countries
- movie_directors
- movie_genres
- movie_locations
- movie_tags
- tags
- user_ratedmovies
- user_taggedmovies

Raw data is stored in `data/raw/`, while processed artifacts are stored in `data/processed/`.

## MLflow

MLflow is used for experiment tracking.

Tracked information includes:

- model parameters
- training metrics
- evaluation metrics
- experiment names
- run metadata

This makes it possible to compare:

- different model types
- different hyperparameter settings
- different preprocessing variants

### Open the MLflow UI

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

## Notebooks

The `notebooks/` directory contains notebooks for:

- EDA
- feature engineering
- model experiments
- explainability workflows

## Scripts

Main experiment runner:

```bash
python scripts/run_experiments.py
```

## Tests

The repository includes tests for:

- the popularity recommender
- collaborative filtering models

Run them with:

```bash
pytest
```

## Coding Conventions

The project aims to keep the codebase:

- readable
- modular
- reproducible
- easy to review
- consistent with the current package layout

Recommended practices:

- keep feature engineering deterministic where possible
- separate data loading from model logic
- keep model evaluation explicit
- avoid mixing notebook-only code with production modules
- keep experiment logging consistent across models

## Reproducibility Notes

To keep experiments reproducible:

- use fixed random seeds when applicable
- log hyperparameters and metrics to MLflow
- keep feature generation deterministic
- version changes through pull requests
- avoid hidden notebook state when comparing runs

## Contributing

When contributing:

- follow the existing project structure
- keep changes scoped to one task
- update tests or notebooks when relevant
- document new scripts or workflows
- prefer clarity over cleverness
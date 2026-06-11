# Explainable Recsys

Explainable Recsys is a movie recommendation system project built around the HetRec-2011 MovieLens dataset. The repository combines exploratory data analysis, feature engineering, multiple recommendation approaches, and explanation methods to make recommendations more interpretable.

## License

This project is licensed under the MIT License.

## Authors

- Farah Abid
- Mateusz Skrzątek
- Michał Raczkiewicz
- Krzysztof Bachanek

## Project Overview

The project includes:

- Exploratory data analysis for the raw dataset tables
- Feature engineering for users, movies, and interactions
- Content-based recommendation
- Collaborative filtering models
- A simple popularity-based baseline
- Explanation-oriented tooling for model interpretation
- MLflow tracking for experiment logging and comparison

## Project Goals

The main goal of the project is to build a recommender system that is not only useful but also explainable.

The system is designed to:

- Compare different recommendation strategies
- Support model evaluation and experimentation
- Produce human-readable features for interpretability tools such as SHAP and LIME
- Keep the development workflow reproducible and review-friendly

## Dataset

The project is based on the HetRec-2011 MovieLens dataset.

### Tables Used

- `movies`
- `movie_actors`
- `movie_countries`
- `movie_directors`
- `movie_genres`
- `movie_locations`
- `movie_tags`
- `tags`
- `user_ratedmovies`
- `user_taggedmovies`

The repository is structured so the raw data is processed into reusable feature tables for downstream modeling and evaluation.

## Repository Structure

```
.
├── data/
├── notebooks/
├── src/
│   ├── content_based/
│   ├── eda/
│   ├── features/
│   └── models/
├── tests/
├── run_experiments.py
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

## Main Modules

### `src/eda/`

- Dataset loading
- Data quality checks
- Table-specific validation and inspection

### `src/features/`

- Feature engineering for users, movies, and interactions
- Reusable aggregations and derived variables

### `src/content_based/`

- Content-based recommendation logic
- Embeddings and explainability utilities

### `src/models/`

- Popularity baseline
- Collaborative filtering models
- MLflow tracking in model training and evaluation

### `notebooks/`

- EDA and research notebooks
- Model exploration
- Interpretability experiments

## Models

The repository currently supports multiple recommendation approaches.

### Popularity-Based Recommender

A simple baseline that ranks movies using weighted popularity and rating quality.

### Collaborative Filtering

Matrix factorization-based recommenders:

- Truncated SVD
- SGD-based matrix factorization

### Content-Based Methods

Feature-driven recommendation using movie content and embeddings.

This makes it possible to compare:

- Baseline quality
- Personalization quality
- Interpretability trade-offs
- Reproducibility across experiments

## Explainability

The project is designed to remain explainable at the feature level.

Typical interpretability support includes:

- Human-readable feature names
- Feature importance analysis
- SHAP-based explanations
- LIME-based explanations
- Traceable experiment logging

The feature engineering layer is intentionally built to keep derived variables understandable and reusable across models.

## Experiment Tracking

MLflow is used for experiment tracking.

Tracked information includes:

- Model parameters
- Evaluation metrics
- Training metrics
- Experiment names
- Run metadata

This allows comparison between:

- Different model types
- Different hyperparameter settings
- Different preprocessing variants

### Inspect Logged Runs

```bash
mlflow ui
```

Then open:

```text
http://localhost:5000
```

## Development Workflow

The project follows a feature-branch and pull-request workflow.

### 1. Issue-Based Work

- Create a GitHub Issue for each task or feature.
- Define the scope clearly.
- Add acceptance criteria when possible.

### 2. Branching

- Create a branch for the task.
- Keep changes focused and small.
- Use descriptive branch names.

Suggested branch types:

```text
feature/...
fix/...
chore/...
experiment/...
```

### 3. Implementation

- Make changes in the branch only.
- Keep code modular and readable.
- Prefer small, reviewable commits.

### 4. Pull Request

- Open a PR when the change is ready.
- Link the related issue.
- Include a concise summary of the change.
- Mention any trade-offs or known limitations.

### 5. Review

- Address review comments directly.
- Keep discussion focused on correctness, maintainability, and clarity.
- Do not merge until checks pass.

### 6. Merge

- Merge only after approval and successful validation.
- Keep history clean and easy to follow.

## Environment Setup

### 1. Create a Virtual Environment

```bash
python -m venv .venv
```

### 2. Activate the Environment

#### Windows

```bash
.venv\Scripts\activate
```

#### macOS / Linux

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

For development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Running the Project

### Run Notebooks

Open notebooks from the `notebooks/` directory to explore:

- EDA
- Feature engineering
- Model experiments
- Explainability workflows

### Run Experiment Script

```bash
python run_experiments.py
```

If you use MLflow tracking, start the UI in a separate terminal:

```bash
mlflow ui
```

## Testing

Run the test suite with:

```bash
pytest
```

If linting is configured in your environment, run Ruff as well:

```bash
ruff check .
```

## Coding Conventions

The project aims to keep the codebase:

- Readable
- Modular
- Reproducible
- Easy to review
- Consistent with the current package layout

### Preferred Practices

- Keep feature engineering deterministic where possible
- Separate data loading from model logic
- Keep model evaluation explicit
- Avoid mixing notebook-only code with production modules
- Keep experiment logging consistent across models

## Reproducibility Notes

To keep experiments reproducible:

- Use fixed random seeds when applicable
- Log hyperparameters and metrics to MLflow
- Keep feature generation deterministic
- Version changes through pull requests
- Avoid hidden notebook state when comparing runs

## Contributing

When contributing:

- Follow the existing project structure
- Keep changes scoped to one task
- Update tests or notebooks when relevant
- Document new scripts or workflows
- Prefer clarity over cleverness

## Status

This repository is actively developed and structured around:

- Explainable recommendation research
- Experimental model comparison
- Reusable feature engineering
- MLflow-based tracking
- Review-friendly development workflow

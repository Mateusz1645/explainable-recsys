import argparse
import json
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from run_experiments import temporal_train_test_split
from src.models.collaborative_filtering.collaborative_filtering import SGDMatrixFactorization, SVDCollaborativeFiltering

DATA_DIR = ROOT / "data"
ARTIFACTS_DIR = ROOT / "artifacts"

def load_interactions(max_rows: int | None):
    cols = ["userID", "movieID", "interaction_rating", "timestamp"]
    df = pd.read_parquet(DATA_DIR / "processed" / "interaction_features.parquet", columns=cols)
    if max_rows is not None and len(df) > max_rows:
        df = df.sort_values("timestamp").tail(max_rows).reset_index(drop=True)
    return df


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-rows", type=int, default=300000)
    parser.add_argument("--svd-factors", type=int, default=50)
    parser.add_argument("--sgd-factors", type=int, default=50)
    parser.add_argument("--sgd-lr", type=float, default=0.005)
    parser.add_argument("--sgd-reg", type=float, default=0.02)
    parser.add_argument("--sgd-epochs", type=int, default=20)
    args = parser.parse_args()

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    interactions = load_interactions(args.max_rows)
    train_df, test_df = temporal_train_test_split(interactions, test_fraction=0.2)

    svd = SVDCollaborativeFiltering(n_factors=args.svd_factors, random_state=42)
    svd.fit(train_df)
    svd_metrics = svd.evaluate(test_df)

    sgd = SGDMatrixFactorization(
        n_factors=args.sgd_factors,
        learning_rate=args.sgd_lr,
        regularization=args.sgd_reg,
        epochs=args.sgd_epochs,
        random_state=42,
    )
    sgd.fit(train_df)
    sgd_metrics = sgd.evaluate(test_df)

    with open(ARTIFACTS_DIR / "svd_model.pkl", "wb") as f:
        pickle.dump(svd, f)

    with open(ARTIFACTS_DIR / "sgd_model.pkl", "wb") as f:
        pickle.dump(sgd, f)

    report = {
        "rows_used": int(len(interactions)),
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "svd_metrics": svd_metrics,
        "sgd_metrics": sgd_metrics,
    }

    with open(ARTIFACTS_DIR / "model_metrics.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Saved:")
    print(str(ARTIFACTS_DIR / "svd_model.pkl"))
    print(str(ARTIFACTS_DIR / "sgd_model.pkl"))
    print(str(ARTIFACTS_DIR / "model_metrics.json"))


if __name__ == "__main__":
    main()
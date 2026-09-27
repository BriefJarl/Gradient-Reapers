from __future__ import annotations

"""
Supervised Model Training & Threshold Optimization for Amazon ML Challenge 2026.

Trains a gradient-boosted decision tree (LightGBM / XGBoost) on pair-level features.
Includes:
- Memory-efficient data loading via DuckDB directly into numpy/pyarrow
- Stratified / ratio-based negative sampling to maintain balanced training
- Early stopping against validation split
- Feature importance extraction
- Automatic decision threshold search maximizing official Macro F_0.5
- Clean persistence of model artifacts and metadata
"""

import argparse
import json
import sys
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np
import pandas as pd
import lightgbm as lgb

from src.features.pair_features import feature_column_names
from src.evaluation.threshold import find_optimal_threshold


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "artifacts" / "features"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
MODEL_DIR = ROOT / "artifacts" / "models"

TRAIN_SPLIT = FEATURE_DIR / "train_split.parquet"
VALID_SPLIT = FEATURE_DIR / "valid_split.parquet"
VALID_S1 = FEATURE_DIR / "valid_s1_entities.parquet"
GROUND_TRUTH = BLOCKING_DIR / "ground_truth_pairs.parquet"

MODEL_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# FEATURE SELECTION
# ============================================================

# Columns in feature_column_names() that are numerical features for GBDT
EXCLUDE_COLS = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "blocking_methods",
    "label",
    "is_match",
}

def get_training_feature_names() -> list[str]:
    """Return the list of numerical feature column names to feed to the model."""
    all_cols = feature_column_names()
    return [col for col in all_cols if col not in EXCLUDE_COLS]


# ============================================================
# DATA LOADER
# ============================================================

def load_training_data(
    con: duckdb.DuckDBPyConnection,
    split_path: Path,
    feature_cols: list[str],
    max_positives: int | None = None,
    neg_to_pos_ratio: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load training data using DuckDB with balanced negative sampling.
    """
    path_sql = str(split_path).replace("\\", "/").replace("'", "''")
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)

    print(f"Sampling training data from: {split_path}")
    print(f"Negative-to-positive ratio: {neg_to_pos_ratio:.1f}")

    total_pos = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path_sql}') WHERE label = 1"
    ).fetchone()[0]

    n_pos = min(total_pos, max_positives) if max_positives else total_pos
    n_neg = int(n_pos * neg_to_pos_ratio)

    sample_query = f"""
    (
        SELECT {feature_list_sql}, label
        FROM read_parquet('{path_sql}')
        WHERE label = 1
        LIMIT {n_pos}
    )
    UNION ALL
    (
        SELECT {feature_list_sql}, label
        FROM read_parquet('{path_sql}')
        WHERE label = 0
        USING SAMPLE {n_neg} ROWS
    )
    """

    df = con.execute(sample_query).fetchdf()

    y = df["label"].to_numpy(dtype=np.int8)
    X = df[feature_cols].to_numpy(dtype=np.float32)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print(f"Loaded training sample: {len(y):,} rows ({n_pos:,} pos, {n_neg:,} neg, {n_pos/len(y)*100:.2f}% positive rate)\n")

    return X, y


def load_validation_sample(
    con: duckdb.DuckDBPyConnection,
    split_path: Path,
    feature_cols: list[str],
    max_positives: int = 300_000,
    neg_to_pos_ratio: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a representative sample of validation pairs for LightGBM early stopping.
    Keeps memory usage small (< 200MB) and evaluation fast (< 0.2s / iteration).
    """
    path_sql = str(split_path).replace("\\", "/").replace("'", "''")
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)

    print(f"Sampling validation monitoring data from: {split_path}")
    total_pos = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path_sql}') WHERE label = 1"
    ).fetchone()[0]

    n_pos = min(total_pos, max_positives)
    n_neg = int(n_pos * neg_to_pos_ratio)

    sample_query = f"""
    (
        SELECT {feature_list_sql}, label
        FROM read_parquet('{path_sql}')
        WHERE label = 1
        LIMIT {n_pos}
    )
    UNION ALL
    (
        SELECT {feature_list_sql}, label
        FROM read_parquet('{path_sql}')
        WHERE label = 0
        USING SAMPLE {n_neg} ROWS
    )
    """

    df = con.execute(sample_query).fetchdf()
    y = df["label"].to_numpy(dtype=np.int8)
    X = df[feature_cols].to_numpy(dtype=np.float32)

    n_pos = int(np.sum(y == 1))
    n_neg = int(np.sum(y == 0))
    print(f"Loaded validation sample: {len(y):,} rows ({n_pos:,} pos, {n_neg:,} neg, {n_pos/len(y)*100:.2f}% positive rate)\n")

    return X, y


def score_validation_set(
    con: duckdb.DuckDBPyConnection,
    booster: lgb.Booster,
    split_path: Path,
    feature_cols: list[str],
    output_path: Path,
    batch_size: int = 1_000_000,
) -> Path:
    """
    Score validation candidate pairs in streaming batches of 1M rows.
    Writes (source1_entity_id, matched_entity_id, probability) to parquet.
    Guarantees peak RAM usage < 1.5GB regardless of split size.
    """
    path_sql = str(split_path).replace("\\", "/").replace("'", "''")
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)
    total_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{path_sql}')").fetchone()[0]
    num_batches = (total_rows + batch_size - 1) // batch_size
    print(f"\nScoring {total_rows:,} validation pairs in {num_batches} batches...")

    tmp_dir = split_path.parent / "tmp_scored_valid"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    parts = []

    for b in range(num_batches):
        offset = b * batch_size
        print(f"  Valid batch {b+1}/{num_batches} (offset {offset:,})...")
        batch_df = con.execute(f"""
            SELECT source1_entity_id, matched_entity_id, {feature_list_sql}
            FROM read_parquet('{path_sql}')
            LIMIT {batch_size} OFFSET {offset}
        """).fetchdf()

        X_b = batch_df[feature_cols].to_numpy(dtype=np.float32)
        batch_df["probability"] = booster.predict(X_b, num_iteration=booster.best_iteration)

        part_file = tmp_dir / f"val_part_{b}.parquet"
        batch_df[["source1_entity_id", "matched_entity_id", "probability"]].to_parquet(part_file, index=False)
        parts.append(str(part_file).replace("\\", "/").replace("'", "''"))

    out_sql = str(output_path).replace("\\", "/").replace("'", "''")
    parts_sql = ", ".join(f"'{p}'" for p in parts)
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet([{parts_sql}])
        ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 500000);
    """)

    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"Saved scored validation pairs to: {output_path}")
    return output_path


# ============================================================
# LIGHTGBM MODEL TRAINING
# ============================================================

def train_lightgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    feature_names: list[str],
    n_estimators: int = 800,
    learning_rate: float = 0.05,
    num_leaves: int = 63,
) -> lgb.Booster:
    """Train a LightGBM binary classifier with early stopping."""
    print("=" * 80)
    print("TRAINING LIGHTGBM CLASSIFIER")
    print("=" * 80)
    print(f"Features: {len(feature_names)}")
    print(f"Trees: {n_estimators} | LR: {learning_rate} | Leaves: {num_leaves}")

    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names, free_raw_data=False)
    dval = lgb.Dataset(X_valid, label=y_valid, feature_name=feature_names, reference=dtrain, free_raw_data=False)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "learning_rate": learning_rate,
        "num_leaves": num_leaves,
        "max_depth": -1,
        "min_child_samples": 50,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "n_jobs": 8,
        "verbose": -1,
        "random_state": 42,
    }

    callbacks = [
        lgb.early_stopping(stopping_rounds=40, verbose=True),
        lgb.log_evaluation(period=50),
    ]

    start_time = time.time()
    booster = lgb.train(
        params=params,
        train_set=dtrain,
        num_boost_round=n_estimators,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=callbacks,
    )
    elapsed = time.time() - start_time
    print(f"\nTraining completed in {elapsed:.1f} seconds. Best iteration: {booster.best_iteration}")

    return booster


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    parser = argparse.ArgumentParser(description="Train Entity Resolution GBDT Ranker.")
    parser.add_argument("--model", choices=["lightgbm"], default="lightgbm", help="Model type to train.")
    parser.add_argument("--max-positives", type=int, default=1_500_000, help="Max positive pairs to sample for training.")
    parser.add_argument("--neg-ratio", type=float, default=3.0, help="Negative-to-positive ratio in training sample.")
    parser.add_argument("--n-estimators", type=int, default=800, help="Max boost rounds.")
    parser.add_argument("--learning-rate", type=float, default=0.05, help="Learning rate.")
    parser.add_argument("--num-leaves", type=int, default=63, help="Max tree leaves.")

    args = parser.parse_args()

    for p in (TRAIN_SPLIT, VALID_SPLIT, GROUND_TRUTH):
        if not p.exists():
            raise FileNotFoundError(f"Missing required artifact:\n{p}")

    feature_cols = get_training_feature_names()
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026 - MODEL TRAINING")
    print("=" * 80)
    print(f"Model Architecture : {args.model.upper()}")
    print(f"Feature count      : {len(feature_cols)}")

    con = duckdb.connect()

    try:
        # Load train and validation data
        X_train, y_train = load_training_data(
            con=con,
            split_path=TRAIN_SPLIT,
            feature_cols=feature_cols,
            max_positives=args.max_positives,
            neg_to_pos_ratio=args.neg_ratio,
        )

        X_valid, y_valid = load_validation_sample(
            con=con,
            split_path=VALID_SPLIT,
            feature_cols=feature_cols,
            max_positives=300_000,
            neg_to_pos_ratio=args.neg_ratio,
        )

        # Train model
        booster = train_lightgbm(
            X_train=X_train,
            y_train=y_train,
            X_valid=X_valid,
            y_valid=y_valid,
            feature_names=feature_cols,
            n_estimators=args.n_estimators,
            learning_rate=args.learning_rate,
            num_leaves=args.num_leaves,
        )

        # Score full validation set in memory-safe batches
        scored_val_path = MODEL_DIR / "scored_valid_pairs.parquet"
        score_validation_set(
            con=con,
            booster=booster,
            split_path=VALID_SPLIT,
            feature_cols=feature_cols,
            output_path=scored_val_path,
        )

        # Optimize Decision Threshold for official Macro F_0.5
        threshold_json_path = MODEL_DIR / "optimal_threshold.json"
        s1_eval_path = VALID_S1 if VALID_S1.exists() else VALID_SPLIT
        print(f"Evaluating validation threshold against S1 universe: {s1_eval_path}")

        thresh_summary = find_optimal_threshold(
            con=con,
            scored_pairs_path=scored_val_path,
            ground_truth_path=GROUND_TRUTH,
            s1_split_path=s1_eval_path,
            output_json=threshold_json_path,
        )

        # Save model and feature importances
        model_path = MODEL_DIR / f"{args.model}_model.txt"
        booster.save_model(str(model_path))
        print(f"\nModel booster saved to: {model_path}")

        # Save feature importance
        importances = booster.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({
            "feature": feature_cols,
            "gain": importances,
        }).sort_values(by="gain", ascending=False)

        imp_path = MODEL_DIR / "feature_importance.csv"
        imp_df.to_csv(imp_path, index=False)
        print(f"Feature importance saved to: {imp_path}")
        print("\nTop 15 Features by Information Gain:")
        print(imp_df.head(15).to_string(index=False))

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("MODEL TRAINING & EVALUATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

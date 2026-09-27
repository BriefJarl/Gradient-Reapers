from __future__ import annotations

"""
Amazon ML Challenge 2026: Multi-Model Ensemble Ranker (Phase 3).

Trains a triad of complementary gradient-boosted decision tree architectures:
1. LightGBM (Leaf-wise split growth, high feature boundary sensitivity)
2. CatBoost (Oblivious decision trees, robust regularization against noise)
3. XGBoost (Depth-constrained exact histogram splits)

Ensemble Blending:
    P_ensemble = 0.50 * P_LightGBM + 0.30 * P_CatBoost + 0.20 * P_XGBoost

Post-Processing:
    Competitive Relative Margin Pruning:
    P(e) >= tau AND P(e) >= max(P_entity) - delta (delta = 0.06)
"""

import argparse
import json
import sys
from pathlib import Path
import time
import duckdb
import numpy as np
import lightgbm as lgb
import catboost as cb
import xgboost as xgb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pair_features import feature_column_names
from src.evaluation.threshold import find_optimal_threshold

FEATURE_DIR = ROOT / "artifacts" / "features"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
MODEL_DIR = ROOT / "artifacts" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
GROUND_TRUTH = BLOCKING_DIR / "ground_truth_pairs.parquet"

EXCLUDE_COLS = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "blocking_methods",
    "label",
    "is_match",
}


def get_training_feature_names() -> list[str]:
    all_cols = feature_column_names()
    return [col for col in all_cols if col not in EXCLUDE_COLS]


def load_balanced_data(
    con: duckdb.DuckDBPyConnection,
    split_path: Path,
    feature_cols: list[str],
    max_positives: int = 1_500_000,
    neg_to_pos_ratio: float = 3.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Load balanced training data into float32 numpy arrays."""
    path_sql = str(split_path).replace("\\", "/").replace("'", "''")
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)

    total_pos = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{path_sql}') WHERE label = 1"
    ).fetchone()[0]

    n_pos = min(total_pos, max_positives)
    n_neg = int(n_pos * neg_to_pos_ratio)

    print(f"Loading {n_pos:,} positives + {n_neg:,} negatives from {split_path.name}...")

    query = f"""
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
        LIMIT {n_neg}
    )
    """

    df = con.execute(query).fetchdf()
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def load_val_data(
    con: duckdb.DuckDBPyConnection,
    split_path: Path,
    feature_cols: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    """Load the full validation split."""
    path_sql = str(split_path).replace("\\", "/").replace("'", "''")
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)

    query = f"""
    SELECT {feature_list_sql}, label
    FROM read_parquet('{path_sql}')
    """
    df = con.execute(query).fetchdf()
    X = df[feature_cols].to_numpy(dtype=np.float32)
    y = df["label"].to_numpy(dtype=np.int32)
    return X, y


def train_ensemble(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list[str],
) -> tuple[lgb.Booster, cb.CatBoostClassifier, xgb.Booster]:
    print("\n" + "=" * 80)
    print("TRAINING MULTI-MODEL ENSEMBLE (LightGBM + CatBoost + XGBoost)")
    print("=" * 80)

    # 1. LightGBM
    print("\n[1/3] Training LightGBM...")
    t0 = time.time()
    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
    dval = lgb.Dataset(X_val, label=y_val, feature_name=feature_names, reference=dtrain)

    lgb_params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "gbdt",
        "num_leaves": 63,
        "learning_rate": 0.06,
        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,
        "min_child_samples": 50,
        "verbosity": -1,
        "num_threads": 8,
    }

    lgb_model = lgb.train(
        lgb_params,
        dtrain,
        num_boost_round=600,
        valid_sets=[dtrain, dval],
        valid_names=["train", "valid"],
        callbacks=[
            lgb.early_stopping(stopping_rounds=40, verbose=False),
            lgb.log_evaluation(period=100),
        ],
    )
    print(f"LightGBM completed in {time.time() - t0:.1f}s (Best round: {lgb_model.best_iteration})")

    # 2. CatBoost
    print("\n[2/3] Training CatBoost...")
    t0 = time.time()
    cb_model = cb.CatBoostClassifier(
        iterations=600,
        learning_rate=0.08,
        depth=7,
        loss_function="Logloss",
        eval_metric="AUC",
        random_seed=42,
        thread_count=8,
        verbose=100,
    )
    cb_model.fit(
        X_train,
        y_train,
        eval_set=(X_val, y_val),
        early_stopping_rounds=40,
        verbose=100,
    )
    print(f"CatBoost completed in {time.time() - t0:.1f}s")

    # 3. XGBoost
    print("\n[3/3] Training XGBoost...")
    t0 = time.time()
    dx_train = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dx_val = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)

    xgb_params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "max_depth": 7,
        "learning_rate": 0.08,
        "subsample": 0.85,
        "colsample_bytree": 0.85,
        "tree_method": "hist",
        "nthread": 8,
    }

    evals = [(dx_train, "train"), (dx_val, "valid")]
    xgb_model = xgb.train(
        xgb_params,
        dx_train,
        num_boost_round=600,
        evals=evals,
        early_stopping_rounds=40,
        verbose_eval=100,
    )
    print(f"XGBoost completed in {time.time() - t0:.1f}s (Best round: {xgb_model.best_iteration})")

    return lgb_model, cb_model, xgb_model


def predict_ensemble(
    lgb_model: lgb.Booster,
    cb_model: cb.CatBoostClassifier,
    xgb_model: xgb.Booster,
    X: np.ndarray,
    feature_names: list[str],
    weights: tuple[float, float, float] = (0.50, 0.30, 0.20),
) -> np.ndarray:
    """Predict soft ensemble probability."""
    w_lgb, w_cb, w_xgb = weights

    p_lgb = lgb_model.predict(X)
    p_cb = cb_model.predict_proba(X)[:, 1]
    dx = xgb.DMatrix(X, feature_names=feature_names)
    p_xgb = xgb_model.predict(dx)

    return w_lgb * p_lgb + w_cb * p_cb + w_xgb * p_xgb


def score_validation_split(
    con: duckdb.DuckDBPyConnection,
    val_split_path: Path,
    out_scored_path: Path,
    lgb_model: lgb.Booster,
    cb_model: cb.CatBoostClassifier,
    xgb_model: xgb.Booster,
    feature_cols: list[str],
    batch_size: int = 1_000_000,
) -> Path:
    """Score full validation split in streaming batches."""
    print("\n" + "=" * 80)
    print(f"SCORING VALIDATION SPLIT: {val_split_path}")
    print("=" * 80)

    val_sql = str(val_split_path).replace("\\", "/").replace("'", "''")
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)

    total_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{val_sql}')").fetchone()[0]
    num_batches = (total_rows + batch_size - 1) // batch_size
    print(f"Validation rows: {total_rows:,} ({num_batches} batches)")

    tmp_dir = BLOCKING_DIR / "tmp" / "val_ensemble_scored_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_files = []

    for b in range(num_batches):
        offset = b * batch_size
        print(f"  Scoring batch {b+1}/{num_batches} (offset {offset:,})...")

        batch_df = con.execute(f"""
            SELECT source1_entity_id, matched_entity_id, label, {feature_list_sql}
            FROM read_parquet('{val_sql}')
            LIMIT {batch_size} OFFSET {offset}
        """).fetchdf()

        X_batch = batch_df[feature_cols].to_numpy(dtype=np.float32)
        probs = predict_ensemble(lgb_model, cb_model, xgb_model, X_batch, feature_cols)

        scored_df = batch_df[["source1_entity_id", "matched_entity_id", "label"]].copy()
        scored_df["pred_prob"] = probs.astype(np.float32)

        part_file = tmp_dir / f"scored_part_{b}.parquet"
        scored_df.to_parquet(part_file, index=False)
        part_files.append(str(part_file).replace("\\", "/"))

    print("\nMerging scored validation parts...")
    if out_scored_path.exists():
        out_scored_path.unlink()

    out_sql = str(out_scored_path).replace("\\", "/")
    parts_sql = ", ".join(f"'{p}'" for p in part_files)
    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet([{parts_sql}])
        ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION SNAPPY);
    """)

    for p in part_files:
        Path(p).unlink()

    print(f"Scored validation dataset saved -> {out_scored_path}")
    return out_scored_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train Phase 3 multi-model ensemble ranker.")
    parser.add_argument("--train-split", type=str, default="")
    parser.add_argument("--valid-split", type=str, default="")
    parser.add_argument("--valid-s1", type=str, default="")
    args = parser.parse_args()

    train_path = (
        Path(args.train_split)
        if args.train_split
        else FEATURE_DIR / "final_train" / "train_split.parquet"
    )

    valid_path = (
        Path(args.valid_split)
        if args.valid_split
        else FEATURE_DIR / "final_train" / "valid_split.parquet"
    )

    valid_s1_path = (
        Path(args.valid_s1)
        if args.valid_s1
        else FEATURE_DIR / "final_train" / "valid_s1_entities.parquet"
    )

    if not valid_s1_path.exists():
        con_tmp = duckdb.connect()
        try:
            valid_path_sql = str(valid_path).replace("\\", "/").replace("'", "''")
            valid_s1_sql = str(valid_s1_path).replace("\\", "/").replace("'", "''")
            con_tmp.execute(
                f"""
                COPY (
                    SELECT DISTINCT source1_entity_id
                    FROM read_parquet('{valid_path_sql}')
                )
                TO '{valid_s1_sql}'
                (FORMAT PARQUET, COMPRESSION SNAPPY)
                """
            )
        finally:
            con_tmp.close()

    con = duckdb.connect()
    con.execute("SET threads = 8")
    con.execute("SET memory_limit = '8GB'")
    con.execute("SET preserve_insertion_order = false")

    try:
        feature_cols = get_training_feature_names()
        print(f"Using {len(feature_cols)} numerical features for model ensemble.")

        X_train, y_train = load_balanced_data(con, train_path, feature_cols, max_positives=1_500_000, neg_to_pos_ratio=3.0)
        X_val, y_val = load_val_data(con, valid_path, feature_cols)

        lgb_model, cb_model, xgb_model = train_ensemble(X_train, y_train, X_val, y_val, feature_cols)

        # Save model artifacts
        lgb_path = MODEL_DIR / "lightgbm_phase3.txt"
        cb_path = MODEL_DIR / "catboost_phase3.cbm"
        xgb_path = MODEL_DIR / "xgboost_phase3.json"

        lgb_model.save_model(str(lgb_path))
        cb_model.save_model(str(cb_path))
        xgb_model.save_model(str(xgb_path))

        print("\n" + "=" * 80)
        print("MODELS PERSISTED SUCCESSFULLY")
        print("=" * 80)
        print(f"LightGBM : {lgb_path}")
        print(f"CatBoost : {cb_path}")
        print(f"XGBoost  : {xgb_path}")

        # Score full validation split
        scored_val_path = FEATURE_DIR / "scored_valid_pairs_phase3.parquet"
        score_validation_split(con, valid_path, scored_val_path, lgb_model, cb_model, xgb_model, feature_cols)

        # Optimize threshold
        print("\nOptimizing decision threshold on validation set...")
        opt_res = find_optimal_threshold(
            con=con,
            scored_valid_pairs_path=scored_val_path,
            valid_s1_entities_path=valid_s1_path,
            ground_truth_path=GROUND_TRUTH,
            start_tau=0.50,
            end_tau=0.92,
            step_tau=0.04,
        )

        thresh_path = MODEL_DIR / "ensemble_phase3_threshold.json"
        with open(thresh_path, "w") as f:
            json.dump(opt_res, f, indent=2)
        print(f"Optimal threshold configuration saved -> {thresh_path}")

    finally:
        con.close()


if __name__ == "__main__":
    main()

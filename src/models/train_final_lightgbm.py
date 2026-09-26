from __future__ import annotations

"""
Production LightGBM trainer for Amazon ML Challenge 2026.

Input:
    artifacts/features/final_train/train_split.parquet
    artifacts/features/final_train/valid_split.parquet

Output:
    artifacts/models/final_lightgbm.txt
    artifacts/models/final_lightgbm_meta.json
    artifacts/features/final_train/valid_predictions.parquet

Design:
- DuckDB for filtering / deterministic negative sampling.
- Numeric-only model matrix; IDs and provenance strings are excluded.
- All positives are retained.
- Negatives are deterministically downsampled to a configurable ratio.
- Memory-aware: chooses a safe negative ratio from available RAM unless
  explicitly overridden.
- Validation predictions are written to Parquet in chunks.
- No pandas is required for the data selection path.
"""

import argparse
import json
import os
from pathlib import Path

import duckdb
import lightgbm as lgb
import numpy as np
import psutil
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "artifacts" / "features" / "final_train"
MODEL_DIR = ROOT / "artifacts" / "models"
TMP_DIR = ROOT / "artifacts" / "blocking" / "duckdb_tmp"

TRAIN = FEATURE_DIR / "train_split.parquet"
VALID = FEATURE_DIR / "valid_split.parquet"
MODEL = MODEL_DIR / "final_lightgbm.txt"
META = MODEL_DIR / "final_lightgbm_meta.json"
VALID_PRED = FEATURE_DIR / "valid_predictions.parquet"

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY", "8GB")
THREADS = int(
    os.environ.get(
        "DUCKDB_THREADS",
        str(max(4, min(12, (os.cpu_count() or 10) - 2))),
    )
)

ID_COLUMNS = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "blocking_methods",
    "label",
}

# These are the canonical numeric features emitted by pair_features.py.
MODEL_FEATURES = [
    "blocking_mask",
    "num_blocking_methods",

    "name_exact",
    "name_compact_exact",
    "name_ascii_exact",
    "name_char_ratio",
    "name_token_overlap",
    "name_token_jaccard",
    "name_length_diff",
    "name_token_count_diff",
    "name_numeric_overlap",
    "name_numeric_exact",
    "name_length_ratio",

    "address_exact",
    "address_compact_exact",
    "address_ascii_exact",
    "address_char_ratio",
    "address_token_overlap",
    "address_token_jaccard",
    "address_length_diff",
    "address_token_count_diff",
    "address_numeric_exact",
    "address_numeric_overlap",
    "address_length_ratio",

    "country_exact",
    "name_present",
    "address_present",

    "block_address",
    "block_address_compact",
    "block_name",
    "block_rare_name",
    "block_rare_address",
    "block_hybrid",
]

DEFAULT_NEG_RATIO = 2.0
VALID_CHUNK = 500_000


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def configure(con: duckdb.DuckDBPyConnection) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{sql_quote(TMP_DIR)}'")


def validate_schema(con: duckdb.DuckDBPyConnection, path: Path) -> None:
    cols = {
        row[0]
        for row in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_quote(path)}')"
        ).fetchall()
    }

    missing = set(MODEL_FEATURES + ["label"]) - cols
    if missing:
        raise RuntimeError(
            f"{path.name}: missing required columns: {sorted(missing)}"
        )


def choose_negative_ratio(requested: float | None) -> float:
    if requested is not None:
        return requested

    available_gb = psutil.virtual_memory().available / (1024**3)

    if available_gb < 12:
        return 1.0
    if available_gb < 20:
        return 1.5
    return DEFAULT_NEG_RATIO


def load_training_matrix(
    con: duckdb.DuckDBPyConnection,
    negative_ratio: float,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    feature_sql = ", ".join(f'"{c}"' for c in MODEL_FEATURES)

    # Count positives first. We retain every positive pair.
    positives = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(TRAIN)}')
            WHERE label = 1
            """
        ).fetchone()[0]
    )

    negatives_to_keep = int(round(positives * negative_ratio))

    total_negatives = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(TRAIN)}')
            WHERE label = 0
            """
        ).fetchone()[0]
    )

    negatives_to_keep = min(negatives_to_keep, total_negatives)

    print()
    print("TRAINING SAMPLE")
    print(f"All positives        : {positives:,}")
    print(f"All negatives        : {total_negatives:,}")
    print(f"Negative ratio       : {negative_ratio:.2f}x")
    print(f"Negatives retained   : {negatives_to_keep:,}")
    print(
        f"Training rows        : "
        f"{positives + negatives_to_keep:,}"
    )

    # Deterministic row-level sampling. The modulo is applied only to
    # negatives; positives are never dropped.
    #
    # The threshold is computed against a 1,000,000 bucket space to avoid
    # floating-point behavior in the SQL predicate.
    threshold = int(
        min(1_000_000, round(
            negatives_to_keep / total_negatives * 1_000_000
        ))
    ) if total_negatives else 0

    query = f"""
        SELECT
            {feature_sql},
            CAST(label AS FLOAT) AS label
        FROM read_parquet('{sql_quote(TRAIN)}')
        WHERE
            label = 1
            OR (
                label = 0
                AND hash(source1_entity_id || '|' || matched_entity_id)
                    % 1000000 < {threshold}
            )
    """

    print()
    print("Loading training matrix...")
    table = con.execute(query).fetch_arrow_table()

    y = table.column("label").to_numpy(zero_copy_only=False).astype(
        np.float32,
        copy=False,
    )

    x_arrays = []
    for name in MODEL_FEATURES:
        arr = table.column(name).to_numpy(zero_copy_only=False)
        x_arrays.append(np.asarray(arr, dtype=np.float32))

    X = np.column_stack(x_arrays).astype(np.float32, copy=False)

    actual_pos = int(np.sum(y == 1))
    actual_neg = int(np.sum(y == 0))

    print(f"Loaded X shape      : {X.shape}")
    print(f"Loaded positives    : {actual_pos:,}")
    print(f"Loaded negatives    : {actual_neg:,}")

    return X, y, actual_pos, actual_neg


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    actual_pos: int,
    actual_neg: int,
) -> lgb.Booster:
    # Because negatives are downsampled, restore the original class balance
    # through sample weighting. This improves probability meaning while the
    # validation threshold is still optimized later.
    original_neg = 37_577_161
    original_pos = 4_405_093

    sample_weight = np.where(
        y == 1,
        1.0,
        (original_neg / original_pos) / (actual_neg / actual_pos),
    ).astype(np.float32)

    train_data = lgb.Dataset(
        X,
        label=y,
        weight=sample_weight,
        feature_name=MODEL_FEATURES,
        free_raw_data=True,
    )

    params = {
        "objective": "binary",
        "metric": ["auc", "binary_logloss"],
        "boosting_type": "gbdt",

        "learning_rate": 0.08,
        "num_leaves": 127,
        "max_depth": -1,
        "min_data_in_leaf": 80,

        "feature_fraction": 0.85,
        "bagging_fraction": 0.85,
        "bagging_freq": 1,

        "lambda_l1": 0.1,
        "lambda_l2": 2.0,
        "min_gain_to_split": 0.0,

        "verbosity": -1,
        "num_threads": THREADS,
        "force_col_wise": True,
        "max_bin": 255,
        "seed": 20260926,
        "feature_fraction_seed": 20260926,
        "bagging_seed": 20260926,
        "data_random_seed": 20260926,
    }

    print()
    print("TRAINING LIGHTGBM")
    print(json.dumps(params, indent=2))

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=700,
    )

    return booster


def write_validation_predictions(
    con: duckdb.DuckDBPyConnection,
    booster: lgb.Booster,
) -> None:
    if VALID_PRED.exists():
        VALID_PRED.unlink()

    feature_sql = ", ".join(f'"{c}"' for c in MODEL_FEATURES)

    count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_quote(VALID)}')"
        ).fetchone()[0]
    )

    print()
    print("SCORING VALIDATION")
    print(f"Validation rows     : {count:,}")

    query = f"""
        SELECT
            source1_entity_id,
            matched_entity_id,
            matched_source,
            label,
            {feature_sql}
        FROM read_parquet('{sql_quote(VALID)}')
    """

    reader = con.execute(query).fetch_record_batch(
        rows_per_batch=VALID_CHUNK
    )

    writer: pq.ParquetWriter | None = None
    scored = 0

    try:
        for batch in reader:
            table = pa.Table.from_batches([batch])

            X = np.column_stack(
                [
                    table.column(name).to_numpy(zero_copy_only=False)
                    for name in MODEL_FEATURES
                ]
            ).astype(np.float32, copy=False)

            pred = booster.predict(X).astype(np.float32, copy=False)

            out_table = pa.table(
                {
                    "source1_entity_id": table.column(
                        "source1_entity_id"
                    ),
                    "matched_entity_id": table.column(
                        "matched_entity_id"
                    ),
                    "matched_source": table.column(
                        "matched_source"
                    ),
                    "label": table.column("label"),
                    "score": pa.array(pred),
                }
            )

            if writer is None:
                writer = pq.ParquetWriter(
                    str(VALID_PRED),
                    out_table.schema,
                    compression="zstd",
                )

            writer.write_table(
                out_table,
                row_group_size=250_000,
            )

            scored += out_table.num_rows
            print(f"  scored {scored:,} / {count:,}")

    finally:
        if writer is not None:
            writer.close()

    if scored != count:
        raise RuntimeError(
            f"Validation scoring row mismatch: {scored:,} != {count:,}"
        )

    print(f"Validation predictions: {VALID_PRED}")

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--negative-ratio",
        type=float,
        default=None,
        help="Negatives per positive. Default is memory-aware (1.0-2.0).",
    )
    args = parser.parse_args()

    if not TRAIN.exists():
        raise FileNotFoundError(TRAIN)
    if not VALID.exists():
        raise FileNotFoundError(VALID)

    negative_ratio = choose_negative_ratio(args.negative_ratio)

    print("=" * 88)
    print("AMAZON ML CHALLENGE 2026")
    print("FINAL LIGHTGBM TRAINER")
    print("=" * 88)
    print(f"Train       : {TRAIN}")
    print(f"Validation  : {VALID}")
    print(f"Model       : {MODEL}")
    print(f"RAM available now: {psutil.virtual_memory().available / (1024**3):.2f} GB")
    print(f"Negative ratio     : {negative_ratio:.2f}x")

    con = duckdb.connect()

    try:
        configure(con)
        validate_schema(con, TRAIN)
        validate_schema(con, VALID)

        X, y, pos, neg = load_training_matrix(
            con,
            negative_ratio,
        )

        booster = train_model(
            X,
            y,
            pos,
            neg,
        )

        if MODEL.exists():
            MODEL.unlink()

        booster.save_model(str(MODEL))

        write_validation_predictions(
            con,
            booster,
        )

        meta = {
            "model": "LightGBM",
            "feature_count": len(MODEL_FEATURES),
            "features": MODEL_FEATURES,
            "negative_ratio": negative_ratio,
            "training_rows": int(len(y)),
            "training_positives": int(pos),
            "training_negatives": int(neg),
            "original_candidate_rows": 41_982_254,
            "original_positive_rows": 4_405_093,
            "original_negative_rows": 37_577_161,
            "seed": 20260926,
        }

        META.write_text(
            json.dumps(meta, indent=2),
            encoding="utf-8",
        )

        print()
        print("=" * 88)
        print("FINAL LIGHTGBM TRAINING PASSED")
        print("=" * 88)
        print(f"MODEL       : {MODEL}")
        print(f"META        : {META}")
        print(f"VALID SCORE : {VALID_PRED}")
        print("=" * 88)

    finally:
        con.close()


if __name__ == "__main__":
    main()

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
- DuckDB for filtering / deterministic negative selection.
- Numeric-only model matrix; IDs and provenance strings are excluded.
- Features are DISCOVERED from the parquet schema, not hardcoded, so that
  feature-engineering work in pair_features.py is picked up automatically.
  The resolved list is persisted to the meta file and is the authoritative
  contract for the scorer.
- All positives are retained.
- Negatives are selected by HARD-NEGATIVE MINING rather than uniform
  downsampling: the highest-similarity non-matches within each
  (source1_entity_id, matched_source) group are kept, plus a smaller uniform
  sample of the easy tail so the model still sees the background distribution.
- Early stopping is driven by an entity-level macro-F0.5 proxy on a held-out
  entity subsample, not by AUC. AUC keeps improving long after the competition
  metric has plateaued.
- Validation predictions are written to Parquet in chunks.
- No pandas is required for the data selection path.

Why hard-negative mining
------------------------
The previous implementation kept a uniform 2:1 hash sample of negatives. With
~19 candidates and ~3.46 true matches per S1 entity, a uniform sample is
dominated by trivially-rejectable pairs, and the decision boundary the model
actually has to get right is under-represented. Keeping the same row budget but
spending it on near-miss negatives is strictly more informative.

Why the sample weighting was removed
------------------------------------
The previous trainer reconstructed the original class balance through sample
weights so that the emitted probabilities were calibrated against the full
candidate pool, which a single global threshold then cut. The decision layer is
now rank-based and per-entity (see src/models/train_decision.py), so absolute
calibration no longer carries information and the weighting only distorted the
loss surface. Pass --scale-pos-weight explicitly if a calibrated model is needed
for some other purpose.
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

# Columns that are never model inputs.
ID_COLUMNS = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "blocking_methods",
    "label",
}

# DuckDB type-name prefixes that are safe to feed to LightGBM.
NUMERIC_PREFIXES = (
    "TINYINT",
    "SMALLINT",
    "INTEGER",
    "BIGINT",
    "HUGEINT",
    "UTINYINT",
    "USMALLINT",
    "UINTEGER",
    "UBIGINT",
    "FLOAT",
    "DOUBLE",
    "DECIMAL",
    "REAL",
    "BOOLEAN",
)

# Similarity columns used to rank negatives by difficulty. Any that are absent
# are skipped and the remaining weights are renormalized, so this survives
# feature-set changes.
PROXY_TERMS: list[tuple[str, float]] = [
    ("name_char_ratio", 0.40),
    ("name_token_jaccard", 0.25),
    ("address_token_jaccard", 0.20),
    ("address_char_ratio", 0.15),
]

DEFAULT_NEG_RATIO = 3.0
DEFAULT_HARD_RANK = 3
DEFAULT_EASY_FRACTION = 0.25
VALID_CHUNK = 500_000

# Entity-level early-stopping proxy. Selection rule is
#   accept if score >= max(T_ABS, ALPHA * best_score_within_entity_and_source)
# This is a fixed stand-in for the real decision layer; it exists only to give
# early stopping a signal that moves with the competition metric.
EVAL_T_ABS = 0.50
EVAL_ALPHA = 0.70


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def ident(column: str) -> str:
    return '"' + column.replace('"', '""') + '"'


def configure(con: duckdb.DuckDBPyConnection) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{sql_quote(TMP_DIR)}'")


def describe(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> dict[str, str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{sql_quote(path)}')"
    ).fetchall()
    return {row[0]: str(row[1]).upper() for row in rows}


def discover_features(schema: dict[str, str]) -> list[str]:
    """
    Resolve the model feature list from the parquet schema.

    Any numeric column that is not an identifier or the label becomes a
    feature. Sorted for a stable column order across runs.
    """
    features = [
        name
        for name, dtype in schema.items()
        if name not in ID_COLUMNS
        and dtype.startswith(NUMERIC_PREFIXES)
    ]

    if not features:
        raise RuntimeError(
            "No numeric feature columns found. "
            f"Schema was: {sorted(schema)}"
        )

    return sorted(features)


def validate_schema(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    features: list[str],
) -> None:
    cols = set(describe(con, path))

    missing = set(features + ["label"]) - cols
    if missing:
        raise RuntimeError(
            f"{path.name}: missing required columns: {sorted(missing)}"
        )


def build_proxy_expr(
    features: set[str],
    alias: str = "t",
) -> str | None:
    """
    Weighted similarity used to order negatives by difficulty.
    """
    terms = [(c, w) for c, w in PROXY_TERMS if c in features]
    if not terms:
        return None

    total = sum(w for _, w in terms)

    return " + ".join(
        f"({w / total:.6f} * COALESCE({alias}.{ident(c)}, 0.0))"
        for c, w in terms
    )


def choose_negative_ratio(requested: float | None) -> float:
    if requested is not None:
        return requested

    available_gb = psutil.virtual_memory().available / (1024**3)

    if available_gb < 12:
        return 1.5
    if available_gb < 20:
        return 2.5
    return DEFAULT_NEG_RATIO


def select_hard_negatives(
    con: duckdb.DuckDBPyConnection,
    proxy_expr: str,
    hard_rank: int,
    budget: int,
) -> int:
    """
    Materialize the keys of the hard negatives into a temp table.

    Only the three identity columns plus the proxy score are sorted, so the
    window function runs over a narrow projection rather than the full feature
    matrix. The wide join happens afterwards.

    Returns the number of retained hard negatives.
    """
    con.execute("DROP TABLE IF EXISTS hard_keys_all")
    con.execute("DROP TABLE IF EXISTS hard_keys")

    con.execute(
        f"""
        CREATE TEMP TABLE hard_keys_all AS
        WITH neg AS (
            SELECT
                t.source1_entity_id,
                t.matched_entity_id,
                t.matched_source,
                {proxy_expr} AS proxy
            FROM read_parquet('{sql_quote(TRAIN)}') t
            WHERE t.label = 0
        ),
        ranked AS (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                ROW_NUMBER() OVER (
                    PARTITION BY source1_entity_id, matched_source
                    ORDER BY
                        proxy DESC,
                        hash(matched_entity_id)
                ) AS rnk
            FROM neg
        )
        SELECT
            source1_entity_id,
            matched_entity_id,
            matched_source
        FROM ranked
        WHERE rnk <= {int(hard_rank)}
        """
    )

    found = int(
        con.execute("SELECT COUNT(*) FROM hard_keys_all").fetchone()[0]
    )

    if found <= budget:
        con.execute(
            "CREATE TEMP TABLE hard_keys AS SELECT * FROM hard_keys_all"
        )
        con.execute("DROP TABLE hard_keys_all")
        return found

    # Deterministically trim to the budget.
    keep_ppm = int(max(0, min(1_000_000, round(budget / found * 1_000_000))))

    con.execute(
        f"""
        CREATE TEMP TABLE hard_keys AS
        SELECT *
        FROM hard_keys_all
        WHERE hash(source1_entity_id || '|' || matched_entity_id)
              % 1000000 < {keep_ppm}
        """
    )
    con.execute("DROP TABLE hard_keys_all")

    return int(con.execute("SELECT COUNT(*) FROM hard_keys").fetchone()[0])


def load_training_matrix(
    con: duckdb.DuckDBPyConnection,
    features: list[str],
    negative_ratio: float,
    hard_rank: int,
    easy_fraction: float,
) -> tuple[np.ndarray, np.ndarray, dict]:
    feature_sql = ", ".join(f"t.{ident(c)}" for c in features)

    positives = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(TRAIN)}')
            WHERE label = 1
            """
        ).fetchone()[0]
    )

    total_negatives = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(TRAIN)}')
            WHERE label = 0
            """
        ).fetchone()[0]
    )

    if positives == 0:
        raise RuntimeError("No positive rows in the training split.")

    budget = min(
        int(round(positives * negative_ratio)),
        total_negatives,
    )

    hard_budget = int(round(budget * (1.0 - easy_fraction)))

    proxy_expr = build_proxy_expr(set(features))

    print()
    print("=" * 88)
    print("NEGATIVE SELECTION")
    print("=" * 88)
    print(f"All positives        : {positives:,}")
    print(f"All negatives        : {total_negatives:,}")
    print(f"Negative ratio       : {negative_ratio:.2f}x")
    print(f"Negative budget      : {budget:,}")

    if proxy_expr is None:
        print()
        print("WARNING: none of the similarity columns used for hardness")
        print("ranking are present. Falling back to uniform sampling.")
        hard_kept = 0
    else:
        print(f"Hard budget          : {hard_budget:,}")
        print(f"Hard rank cutoff     : top {hard_rank} per (entity, source)")
        print()
        print("Ranking negatives by similarity proxy...")
        hard_kept = select_hard_negatives(
            con,
            proxy_expr,
            hard_rank,
            hard_budget,
        )
        print(f"Hard negatives kept  : {hard_kept:,}")

    easy_budget = max(0, budget - hard_kept)
    easy_pool = max(1, total_negatives - hard_kept)
    easy_ppm = int(
        max(0, min(1_000_000, round(easy_budget / easy_pool * 1_000_000)))
    )

    print(f"Easy negatives target: {easy_budget:,}")
    print(f"Training rows (est.) : {positives + hard_kept + easy_budget:,}")

    if hard_kept > 0:
        query = f"""
            SELECT
                {feature_sql},
                CAST(t.label AS FLOAT) AS label
            FROM read_parquet('{sql_quote(TRAIN)}') t
            LEFT JOIN hard_keys h
                   ON t.source1_entity_id = h.source1_entity_id
                  AND t.matched_entity_id = h.matched_entity_id
                  AND t.matched_source    = h.matched_source
            WHERE
                t.label = 1
                OR h.source1_entity_id IS NOT NULL
                OR (
                    t.label = 0
                    AND hash(
                        t.source1_entity_id || '|' || t.matched_entity_id
                    ) % 1000000 < {easy_ppm}
                )
        """
    else:
        query = f"""
            SELECT
                {feature_sql},
                CAST(t.label AS FLOAT) AS label
            FROM read_parquet('{sql_quote(TRAIN)}') t
            WHERE
                t.label = 1
                OR (
                    t.label = 0
                    AND hash(
                        t.source1_entity_id || '|' || t.matched_entity_id
                    ) % 1000000 < {easy_ppm}
                )
        """

    print()
    print("Loading training matrix...")
    table = con.execute(query).fetch_arrow_table()

    y = np.asarray(
        table.column("label").to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )

    X = np.column_stack(
        [
            np.asarray(
                table.column(name).to_numpy(zero_copy_only=False),
                dtype=np.float32,
            )
            for name in features
        ]
    ).astype(np.float32, copy=False)

    del table

    actual_pos = int(np.sum(y == 1))
    actual_neg = int(np.sum(y == 0))

    print(f"Loaded X shape       : {X.shape}")
    print(f"Loaded positives     : {actual_pos:,}")
    print(f"Loaded negatives     : {actual_neg:,}")

    stats = {
        "train_split_positives": positives,
        "train_split_negatives": total_negatives,
        "negative_ratio": negative_ratio,
        "negative_budget": budget,
        "hard_rank": hard_rank,
        "easy_fraction": easy_fraction,
        "hard_negatives_kept": hard_kept,
        "easy_negatives_target": easy_budget,
        "training_rows": int(len(y)),
        "training_positives": actual_pos,
        "training_negatives": actual_neg,
    }

    return X, y, stats


def load_eval_subsample(
    con: duckdb.DuckDBPyConnection,
    features: list[str],
    entity_percent: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load every candidate row for a deterministic subsample of validation
    entities. Whole entities are taken so the entity-level metric is exact on
    the entities it covers.

    DuckDB emits a dense contiguous group id so the numpy side needs no
    string handling.
    """
    feature_sql = ", ".join(ident(c) for c in features)

    query = f"""
        WITH sub AS (
            SELECT *
            FROM read_parquet('{sql_quote(VALID)}')
            WHERE hash(source1_entity_id) % 100 < {int(entity_percent)}
        )
        SELECT
            {feature_sql},
            CAST(label AS FLOAT) AS label,
            DENSE_RANK() OVER (
                ORDER BY source1_entity_id, matched_source
            ) AS gid
        FROM sub
        ORDER BY gid
    """

    print()
    print(
        f"Loading early-stopping subsample "
        f"({entity_percent}% of validation entities)..."
    )

    table = con.execute(query).fetch_arrow_table()

    y = np.asarray(
        table.column("label").to_numpy(zero_copy_only=False),
        dtype=np.float32,
    )
    gid = np.asarray(
        table.column("gid").to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )

    X = np.column_stack(
        [
            np.asarray(
                table.column(name).to_numpy(zero_copy_only=False),
                dtype=np.float32,
            )
            for name in features
        ]
    ).astype(np.float32, copy=False)

    del table

    print(f"Subsample rows       : {len(y):,}")
    print(f"Subsample groups     : {len(np.unique(gid)):,}")

    return X, y, gid


class EntityF05Metric:
    """
    Macro-F0.5 over (entity, source) groups under a fixed relative-threshold
    selection rule.

    This mirrors the semantics of src/evaluation/evaluate_entity_f05.py: a
    group with no true matches scores 1.0 when nothing is predicted for it.
    It is a proxy for the real decision layer and is used only to stop
    training at a sensible point.
    """

    def __init__(
        self,
        gid: np.ndarray,
        label: np.ndarray,
        t_abs: float,
        alpha: float,
    ) -> None:
        if gid.size == 0:
            raise ValueError("Empty evaluation subsample.")

        # gid arrives sorted ascending, so groups are contiguous.
        boundary = np.empty(gid.size, dtype=bool)
        boundary[0] = True
        np.not_equal(gid[1:], gid[:-1], out=boundary[1:])

        self.starts = np.flatnonzero(boundary)
        self.counts = np.diff(
            np.append(self.starts, gid.size)
        ).astype(np.int64)

        self.label = (label > 0.5)
        self.true_count = np.add.reduceat(
            self.label.astype(np.float64),
            self.starts,
        )
        self.t_abs = float(t_abs)
        self.alpha = float(alpha)

    def __call__(self, preds, eval_data):
        preds = np.asarray(preds, dtype=np.float64).ravel()

        # LightGBM hands builtin binary objectives probabilities, but be
        # defensive: the relative rule is not invariant under the sigmoid.
        if preds.min() < 0.0 or preds.max() > 1.0:
            preds = 1.0 / (1.0 + np.exp(-preds))

        group_max = np.maximum.reduceat(preds, self.starts)
        row_threshold = np.repeat(
            np.maximum(self.t_abs, self.alpha * group_max),
            self.counts,
        )

        accepted = preds >= row_threshold

        pred_count = np.add.reduceat(
            accepted.astype(np.float64),
            self.starts,
        )
        tp = np.add.reduceat(
            (accepted & self.label).astype(np.float64),
            self.starts,
        )

        precision = np.where(
            pred_count > 0,
            tp / np.maximum(pred_count, 1.0),
            0.0,
        )
        recall = np.where(
            self.true_count > 0,
            tp / np.maximum(self.true_count, 1.0),
            0.0,
        )

        denom = 0.25 * precision + recall
        f05 = np.where(
            denom > 0,
            1.25 * precision * recall / np.maximum(denom, 1e-12),
            0.0,
        )
        f05 = np.where(
            (pred_count == 0) & (self.true_count == 0),
            1.0,
            f05,
        )

        return "macro_f05_proxy", float(f05.mean()), True


def train_model(
    X: np.ndarray,
    y: np.ndarray,
    features: list[str],
    eval_bundle: tuple[np.ndarray, np.ndarray, np.ndarray] | None,
    num_boost_round: int,
    early_stopping: int,
    learning_rate: float,
    scale_pos_weight: float | None,
) -> tuple[lgb.Booster, dict]:
    train_data = lgb.Dataset(
        X,
        label=y,
        feature_name=features,
        free_raw_data=True,
    )

    params = {
        "objective": "binary",
        # Early stopping is driven by the custom entity-level metric only.
        "metric": "None",
        "boosting_type": "gbdt",

        "learning_rate": learning_rate,
        "num_leaves": 255,
        "max_depth": -1,
        "min_data_in_leaf": 200,

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

    if scale_pos_weight is not None:
        params["scale_pos_weight"] = scale_pos_weight

    valid_sets = []
    valid_names = []
    feval = None
    callbacks = [lgb.log_evaluation(period=25)]

    if eval_bundle is not None:
        Xv, yv, gid = eval_bundle
        valid_sets = [
            lgb.Dataset(
                Xv,
                label=yv,
                feature_name=features,
                reference=train_data,
                free_raw_data=True,
            )
        ]
        valid_names = ["valid_subsample"]
        feval = EntityF05Metric(gid, yv, EVAL_T_ABS, EVAL_ALPHA)
        callbacks.append(
            lgb.early_stopping(
                stopping_rounds=early_stopping,
                first_metric_only=True,
                verbose=True,
            )
        )

    print()
    print("=" * 88)
    print("TRAINING LIGHTGBM")
    print("=" * 88)
    print(json.dumps(params, indent=2))
    print(f"Max rounds           : {num_boost_round}")
    if eval_bundle is not None:
        print(f"Early stopping       : {early_stopping} rounds")
        print(
            f"Selection proxy      : score >= max("
            f"{EVAL_T_ABS}, {EVAL_ALPHA} * group_best)"
        )
    else:
        print("Early stopping       : disabled (no eval subsample)")

    booster = lgb.train(
        params,
        train_data,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        feval=feval,
        callbacks=callbacks,
    )

    info = {
        "params": {
            k: v for k, v in params.items() if k != "num_threads"
        },
        "num_boost_round_requested": num_boost_round,
        "best_iteration": int(booster.best_iteration or booster.num_trees()),
        "trees": int(booster.num_trees()),
    }

    if booster.best_score:
        try:
            info["best_macro_f05_proxy"] = float(
                booster.best_score["valid_subsample"]["macro_f05_proxy"]
            )
        except (KeyError, TypeError):
            pass

    return booster, info


def write_validation_predictions(
    con: duckdb.DuckDBPyConnection,
    booster: lgb.Booster,
    features: list[str],
) -> None:
    if VALID_PRED.exists():
        VALID_PRED.unlink()

    feature_sql = ", ".join(ident(c) for c in features)

    count = int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_quote(VALID)}')"
        ).fetchone()[0]
    )

    print()
    print("=" * 88)
    print("SCORING FULL VALIDATION SET")
    print("=" * 88)
    print(f"Validation rows      : {count:,}")

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
                    np.asarray(
                        table.column(name).to_numpy(zero_copy_only=False),
                        dtype=np.float32,
                    )
                    for name in features
                ]
            ).astype(np.float32, copy=False)

            pred = booster.predict(
                X,
                num_iteration=booster.best_iteration or None,
            ).astype(np.float32, copy=False)

            out_table = pa.table(
                {
                    "source1_entity_id": table.column("source1_entity_id"),
                    "matched_entity_id": table.column("matched_entity_id"),
                    "matched_source": table.column("matched_source"),
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

            writer.write_table(out_table, row_group_size=250_000)

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
        help="Negatives per positive. Default is memory-aware (1.5-3.0).",
    )
    parser.add_argument(
        "--hard-rank",
        type=int,
        default=DEFAULT_HARD_RANK,
        help=(
            "Keep the top-N hardest negatives per (entity, source). "
            f"Default {DEFAULT_HARD_RANK}."
        ),
    )
    parser.add_argument(
        "--easy-fraction",
        type=float,
        default=DEFAULT_EASY_FRACTION,
        help=(
            "Share of the negative budget spent on a uniform sample of the "
            f"easy tail. Default {DEFAULT_EASY_FRACTION}."
        ),
    )
    parser.add_argument(
        "--num-boost-round",
        type=int,
        default=3000,
        help="Maximum boosting rounds. Early stopping usually ends sooner.",
    )
    parser.add_argument(
        "--early-stopping",
        type=int,
        default=100,
        help="Early-stopping patience in rounds. 0 disables early stopping.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.05,
    )
    parser.add_argument(
        "--eval-entity-percent",
        type=int,
        default=20,
        help=(
            "Percent of validation ENTITIES used for early stopping. "
            "Whole entities are taken. Default 20."
        ),
    )
    parser.add_argument(
        "--scale-pos-weight",
        type=float,
        default=None,
        help=(
            "Optional positive-class weight. Omit for an unweighted fit; "
            "the decision layer is rank-based and does not need calibrated "
            "probabilities."
        ),
    )
    parser.add_argument(
        "--skip-validation-scoring",
        action="store_true",
        help="Train and save the model without scoring the full valid split.",
    )
    args = parser.parse_args()

    if not 0.0 <= args.easy_fraction < 1.0:
        raise ValueError("--easy-fraction must be in [0, 1).")
    if args.hard_rank < 0:
        raise ValueError("--hard-rank must be >= 0.")
    if not 1 <= args.eval_entity_percent <= 100:
        raise ValueError("--eval-entity-percent must be in [1, 100].")

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
    print(
        f"RAM available now: "
        f"{psutil.virtual_memory().available / (1024**3):.2f} GB"
    )

    con = duckdb.connect()

    try:
        configure(con)

        train_schema = describe(con, TRAIN)
        features = discover_features(train_schema)
        validate_schema(con, VALID, features)

        print()
        print(f"Discovered {len(features)} model features:")
        for name in features:
            print(f"  - {name}")

        X, y, stats = load_training_matrix(
            con,
            features,
            negative_ratio,
            args.hard_rank,
            args.easy_fraction,
        )

        eval_bundle = None
        if args.early_stopping > 0:
            eval_bundle = load_eval_subsample(
                con,
                features,
                args.eval_entity_percent,
            )

        booster, train_info = train_model(
            X,
            y,
            features,
            eval_bundle,
            args.num_boost_round,
            args.early_stopping,
            args.learning_rate,
            args.scale_pos_weight,
        )

        del X, y, eval_bundle

        if MODEL.exists():
            MODEL.unlink()

        booster.save_model(
            str(MODEL),
            num_iteration=booster.best_iteration or None,
        )

        if not args.skip_validation_scoring:
            write_validation_predictions(con, booster, features)

        importance = dict(
            zip(
                features,
                (
                    int(v)
                    for v in booster.feature_importance(importance_type="gain")
                ),
            )
        )

        meta = {
            "model": "LightGBM",
            "feature_count": len(features),
            # Authoritative feature contract for the scorer.
            "features": features,
            "seed": 20260926,
            "eval_entity_percent": args.eval_entity_percent,
            "eval_selection_rule": {
                "t_abs": EVAL_T_ABS,
                "alpha": EVAL_ALPHA,
                "note": (
                    "Fixed proxy rule used for early stopping only. The "
                    "production decision layer is train_decision.py."
                ),
            },
            **stats,
            **train_info,
            "feature_importance_gain": dict(
                sorted(
                    importance.items(),
                    key=lambda kv: kv[1],
                    reverse=True,
                )
            ),
        }

        META.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        print()
        print("=" * 88)
        print("FINAL LIGHTGBM TRAINING PASSED")
        print("=" * 88)
        print(f"MODEL       : {MODEL}")
        print(f"META        : {META}")
        print(f"Best iter   : {train_info['best_iteration']:,}")
        if "best_macro_f05_proxy" in train_info:
            print(
                f"Proxy F0.5  : "
                f"{train_info['best_macro_f05_proxy']:.6f}"
            )
        if not args.skip_validation_scoring:
            print(f"VALID SCORE : {VALID_PRED}")
        print("=" * 88)
        print()
        print("Next: python -m src.evaluation.threshold")
        print("      python -m src.models.train_decision")

    finally:
        con.close()


if __name__ == "__main__":
    main()

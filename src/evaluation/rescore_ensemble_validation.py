from __future__ import annotations

import argparse
import sys
from pathlib import Path

import duckdb
import lightgbm as lgb
import catboost as cb
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.features.pair_features import feature_column_names

FEATURES = ROOT / "artifacts" / "features" / "final_train" / "valid_split.parquet"
LGB_MODEL = ROOT / "artifacts" / "models" / "lightgbm_phase3.txt"
CB_MODEL = ROOT / "artifacts" / "models" / "catboost_phase3.cbm"
XGB_MODEL = ROOT / "artifacts" / "models" / "xgboost_phase3.json"
OUT = ROOT / "artifacts" / "features" / "scored_valid_pairs_phase3_fixed.parquet"

THREADS = 8
MEMORY = "6GB"
BATCH_SIZE = 500_000

EXCLUDE = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "blocking_methods",
    "label",
    "is_match",
}


def sql_path(p: Path) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def get_features() -> list[str]:
    return [c for c in feature_column_names() if c not in EXCLUDE]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rescore the full validation split with saved ensemble models."
    )
    parser.add_argument("--input", default=str(FEATURES))
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--w-lgb", type=float, default=0.50)
    parser.add_argument("--w-cb", type=float, default=0.30)
    parser.add_argument("--w-xgb", type=float, default=0.20)
    args = parser.parse_args()

    inp = Path(args.input)
    out = Path(args.output)

    for p in (inp, LGB_MODEL, CB_MODEL, XGB_MODEL):
        if not p.exists():
            raise FileNotFoundError(p)

    weights = np.array([args.w_lgb, args.w_cb, args.w_xgb], dtype=np.float64)
    if np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError("Weights must be non-negative and sum to > 0.")
    weights /= weights.sum()

    feature_cols = get_features()
    feature_sql = ", ".join(f'"{c}"' for c in feature_cols)

    print("=" * 88)
    print("ENSEMBLE VALIDATION RESCORING")
    print("=" * 88)
    print(f"Input       : {inp}")
    print(f"Output      : {out}")
    print(f"Features    : {len(feature_cols)}")
    print(f"Batch size  : {args.batch_size:,}")
    print(f"Weights     : LGB={weights[0]:.4f}, CB={weights[1]:.4f}, XGB={weights[2]:.4f}")

    print("\nLoading saved models...")
    lgb_model = lgb.Booster(model_file=str(LGB_MODEL))
    cb_model = cb.CatBoostClassifier()
    cb_model.load_model(str(CB_MODEL))

    import xgboost as xgb
    xgb_model = xgb.Booster()
    xgb_model.load_model(str(XGB_MODEL))

    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY}'")
    con.execute("SET preserve_insertion_order = false")
    tmp = ROOT / "artifacts" / "blocking" / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{sql_path(tmp)}'")

    total = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_path(inp)}')"
    ).fetchone()[0]

    print(f"Validation rows: {total:,}")

    query = f"""
        SELECT
            source1_entity_id,
            matched_entity_id,
            matched_source,
            label,
            {feature_sql}
        FROM read_parquet('{sql_path(inp)}')
    """

    reader = con.execute(query).fetch_record_batch(rows_per_batch=args.batch_size)

    part_dir = out.parent / "tmp_ensemble_rescore_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    parts = []
    processed = 0

    try:
        for i, batch in enumerate(reader):
            df = batch.to_pandas()

            X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)

            p_lgb = lgb_model.predict(X).astype(np.float32, copy=False)
            p_cb = cb_model.predict_proba(X)[:, 1].astype(np.float32, copy=False)

            dx = xgb.DMatrix(X, feature_names=feature_cols)
            p_xgb = xgb_model.predict(dx).astype(np.float32, copy=False)

            p_ens = (
                weights[0] * p_lgb
                + weights[1] * p_cb
                + weights[2] * p_xgb
            ).astype(np.float32)

            scored = df[
                ["source1_entity_id", "matched_entity_id", "matched_source", "label"]
            ].copy()
            scored["p_lgb"] = p_lgb
            scored["p_cb"] = p_cb
            scored["p_xgb"] = p_xgb
            scored["pred_prob"] = p_ens

            part = part_dir / f"part_{i:04d}.parquet"
            scored.to_parquet(part, index=False, compression="zstd")
            parts.append(str(part).replace("\\", "/"))

            processed += len(df)
            print(f"  scored {processed:,} / {total:,}")

    finally:
        con.close()

    if processed != total:
        raise RuntimeError(f"Scored {processed:,} rows, expected {total:,}.")

    if out.exists():
        out.unlink()

    duck = duckdb.connect()
    duck.execute(f"SET threads = {THREADS}")
    duck.execute(f"SET memory_limit = '{MEMORY}'")
    duck.execute("SET preserve_insertion_order = false")
    duck.execute(f"SET temp_directory = '{sql_path(tmp)}'")

    parts_sql = ", ".join(f"'{p}'" for p in parts)
    duck.execute(f"""
        COPY (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                label,
                p_lgb,
                p_cb,
                p_xgb,
                pred_prob
            FROM read_parquet([{parts_sql}])
        )
        TO '{sql_path(out)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE 250000)
    """)

    # Hard validation: pair identity includes matched_source.
    duplicate_groups = duck.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT source1_entity_id, matched_entity_id, matched_source
            FROM read_parquet('{sql_path(out)}')
            GROUP BY 1,2,3
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if duplicate_groups:
        raise RuntimeError(
            f"Duplicate validation pair groups after rescore: {duplicate_groups:,}"
        )

    print("\n" + "=" * 88)
    print("RESCORING PASSED")
    print("=" * 88)
    print(f"Rows        : {processed:,}")
    print(f"Output      : {out}")
    print("Pair key    : source1_entity_id + matched_entity_id + matched_source")
    print("Models      : saved LightGBM + CatBoost + XGBoost")
    print("=" * 88)

    duck.close()

    for p in parts:
        Path(p).unlink(missing_ok=True)
    try:
        part_dir.rmdir()
    except OSError:
        pass


if __name__ == "__main__":
    main()

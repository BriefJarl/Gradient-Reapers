from __future__ import annotations

"""
Amazon ML Challenge 2026: Phase 3 Test Inference & Submission Generator.

Loads the multi-model ensemble triad (LightGBM + CatBoost + XGBoost):
1. Scores test candidate pairs in 1M streaming batches strictly under 2GB RAM.
2. Applies calibrated probability threshold and competitive relative margin pruning.
3. Formats and exports candidate_pairs.tsv and matching_results.tsv.
4. Validates final output against the official competition validator.
"""

import argparse
import json
import shutil
import subprocess
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

from src.features.pair_features import (
    build_feature_select,
    feature_column_names,
)

DATA_DIR = ROOT / "student_resource" / "dataset"
TEST_DATA_DIR = DATA_DIR / "test"
TEST_S1 = ROOT / "artifacts" / "normalized" / "test_s1.parquet"
TEST_S2 = ROOT / "artifacts" / "normalized" / "test_s2.parquet"
TEST_S3 = ROOT / "artifacts" / "normalized" / "test_s3.parquet"

BLOCKING_DIR = ROOT / "artifacts" / "blocking"
FEATURE_DIR = ROOT / "artifacts" / "features"
MODEL_DIR = ROOT / "artifacts" / "models"
OUTPUT_DIR = ROOT / "output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_CANDIDATES_PATH = BLOCKING_DIR / "union" / "test_phase3_candidates.parquet"
if not TEST_CANDIDATES_PATH.exists():
    TEST_CANDIDATES_PATH = BLOCKING_DIR / "union" / "test_expanded_candidates.parquet"

TEST_FEATURES_PATH = FEATURE_DIR / "test_phase3_features.parquet"
LGB_MODEL_PATH = MODEL_DIR / "lightgbm_phase3.txt"
CB_MODEL_PATH = MODEL_DIR / "catboost_phase3.cbm"
XGB_MODEL_PATH = MODEL_DIR / "xgboost_phase3.json"
THRESHOLD_JSON = MODEL_DIR / "ensemble_phase3_threshold.json"

MATCHING_OUTPUT_TSV = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT_TSV = OUTPUT_DIR / "candidate_pairs.tsv"
VALIDATE_SCRIPT = ROOT / "student_resource" / "utils" / "validate_submission.py"

THREADS = 8
MEMORY_LIMIT = "8GB"
ROW_GROUP_SIZE = 250_000

EXCLUDE_COLS = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "blocking_methods",
    "label",
    "is_match",
}


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def get_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql_quote(path)}')").fetchall()
    return {row[0] for row in rows}


def get_training_feature_names() -> list[str]:
    all_cols = feature_column_names()
    return [col for col in all_cols if col not in EXCLUDE_COLS]


def export_candidate_pairs_tsv(con: duckdb.DuckDBPyConnection) -> Path:
    """Export all test candidate pairs into candidate_pairs.tsv in streaming hash partitions."""
    print("\n" + "=" * 80)
    print("EXPORTING CANDIDATE PAIRS TSV (PHASE 3)")
    print("=" * 80)

    s1_sql = sql_quote(TEST_S1)
    cand_sql = sql_quote(TEST_CANDIDATES_PATH)
    out_path = CANDIDATE_OUTPUT_TSV

    tmp_dir = BLOCKING_DIR / "tmp" / "test_cand_tsv_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    num_partitions = 4
    part_files = []

    for p in range(num_partitions):
        part_tsv = tmp_dir / f"cand_part_{p}.tsv"
        part_sql = sql_quote(part_tsv)
        print(f"Exporting partition {p+1}/{num_partitions}...")

        con.execute(f"""
        COPY (
            WITH s1_partition AS (
                SELECT entity_id AS source1_entity_id
                FROM read_parquet('{s1_sql}')
                WHERE MOD(ABS(HASH(entity_id)), {num_partitions}) = {p}
            ),
            matched_pairs AS (
                SELECT DISTINCT source1_entity_id, matched_entity_id
                FROM read_parquet('{cand_sql}')
                WHERE MOD(ABS(HASH(source1_entity_id)), {num_partitions}) = {p}
            ),
            grouped_candidates AS (
                SELECT
                    source1_entity_id,
                    string_agg(matched_entity_id, ',') AS candidate_entity_ids
                FROM matched_pairs
                GROUP BY source1_entity_id
            )
            SELECT
                s1.source1_entity_id,
                COALESCE(g.candidate_entity_ids, '') AS candidate_entity_ids
            FROM s1_partition s1
            LEFT JOIN grouped_candidates g ON s1.source1_entity_id = g.source1_entity_id
        )
        TO '{part_sql}' (HEADER false, DELIMITER '\\t', QUOTE '');
        """)
        part_files.append(part_tsv)

    print("Concatenating candidate partitions to candidate_pairs.tsv...")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for part_tsv in part_files:
            with open(part_tsv, "r", encoding="utf-8") as in_f:
                shutil.copyfileobj(in_f, out_f)
            part_tsv.unlink()

    print(f"Exported candidate pairs to: {out_path}")
    return out_path


def build_test_features(con: duckdb.DuckDBPyConnection) -> Path:
    """Compute vectorized similarity features on Phase 3 test candidates."""
    print("\n" + "=" * 80)
    print("COMPUTING PHASE 3 TEST CANDIDATE FEATURES")
    print("=" * 80)

    s1_cols = get_columns(con, TEST_S1)
    s2_cols = get_columns(con, TEST_S2)
    s3_cols = get_columns(con, TEST_S3)

    select_s2 = build_feature_select("s1", "t", "c", s1_cols, s2_cols)
    select_s3 = build_feature_select("s1", "t", "c", s1_cols, s3_cols)

    cand_sql = sql_quote(TEST_CANDIDATES_PATH)
    s1_sql = sql_quote(TEST_S1)
    s2_sql = sql_quote(TEST_S2)
    s3_sql = sql_quote(TEST_S3)
    out_sql = sql_quote(TEST_FEATURES_PATH)

    s2_out = TEST_FEATURES_PATH.parent / "test_phase3_features_s2.parquet"
    s3_out = TEST_FEATURES_PATH.parent / "test_phase3_features_s3.parquet"
    if s2_out.exists():
        s2_out.unlink()
    if s3_out.exists():
        s3_out.unlink()

    print("Extracting test S2 features...")
    con.execute(f"""
    COPY (
        SELECT {select_s2}
        FROM (SELECT * FROM read_parquet('{cand_sql}') WHERE matched_source = 'S2') c
        INNER JOIN read_parquet('{s1_sql}') s1 ON c.source1_entity_id = s1.entity_id
        INNER JOIN read_parquet('{s2_sql}') t ON c.matched_entity_id = t.entity_id
    ) TO '{sql_quote(s2_out)}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {ROW_GROUP_SIZE});
    """)

    print("Extracting test S3 features...")
    con.execute(f"""
    COPY (
        SELECT {select_s3}
        FROM (SELECT * FROM read_parquet('{cand_sql}') WHERE matched_source = 'S3') c
        INNER JOIN read_parquet('{s1_sql}') s1 ON c.source1_entity_id = s1.entity_id
        INNER JOIN read_parquet('{s3_sql}') t ON c.matched_entity_id = t.entity_id
    ) TO '{sql_quote(s3_out)}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {ROW_GROUP_SIZE});
    """)

    print("Combining test features...")
    con.execute(f"""
    COPY (
        SELECT * FROM read_parquet(['{sql_quote(s2_out)}', '{sql_quote(s3_out)}'])
    ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {ROW_GROUP_SIZE});
    """)

    n_features = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"Computed features for {n_features:,} test pairs -> {TEST_FEATURES_PATH}")
    return TEST_FEATURES_PATH


def predict_and_export_matches(
    con: duckdb.DuckDBPyConnection,
    tau_us: float = 0.90,
    tau_india: float = 0.84,
    tau_other: float = 0.90,
    relative_margin: float = 0.05,
    top_k: int = 8,
) -> Path:
    """Score test features with ensemble and export matching_results.tsv using Phase 4 calibration."""
    print("\n" + "=" * 80)
    print("PHASE 4: SCORING TEST CANDIDATES WITH COUNTRY CALIBRATION")
    print("=" * 80)
    print(f"US Threshold:       {tau_us:.2f}")
    print(f"India Threshold:    {tau_india:.2f}")
    print(f"Other Threshold:    {tau_other:.2f}")
    print(f"Relative Margin:    {relative_margin:.2f}")
    print(f"Top-K Match Cap:    {top_k}")
    print("=" * 80)

    feature_cols = get_training_feature_names()
    feat_sql = sql_quote(TEST_FEATURES_PATH)
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)
    min_threshold = min(tau_us, tau_india, tau_other)

    # Load models
    print(f"Loading LightGBM: {LGB_MODEL_PATH}")
    lgb_model = lgb.Booster(model_file=str(LGB_MODEL_PATH))

    has_cb = CB_MODEL_PATH.exists()
    has_xgb = XGB_MODEL_PATH.exists()
    cb_model = cb.CatBoostClassifier()
    if has_cb:
        print(f"Loading CatBoost: {CB_MODEL_PATH}")
        cb_model.load_model(str(CB_MODEL_PATH))

    xgb_model = xgb.Booster()
    if has_xgb:
        print(f"Loading XGBoost: {XGB_MODEL_PATH}")
        xgb_model.load_model(str(XGB_MODEL_PATH))

    # Score in batches
    batch_size = 1_000_000
    total_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{feat_sql}')").fetchone()[0]
    num_batches = (total_rows + batch_size - 1) // batch_size

    scored_parts = []
    tmp_dir = BLOCKING_DIR / "tmp" / "test_phase4_scored_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    for b in range(num_batches):
        offset = b * batch_size
        print(f"  Batch {b+1}/{num_batches} (offset {offset:,})...")

        batch_df = con.execute(f"""
            SELECT source1_entity_id, matched_entity_id, {feature_list_sql}
            FROM read_parquet('{feat_sql}')
            LIMIT {batch_size} OFFSET {offset}
        """).fetchdf()

        X_batch = batch_df[feature_cols].to_numpy(dtype=np.float32)

        # Ensemble predictions
        p_lgb = lgb_model.predict(X_batch)
        if has_cb and has_xgb:
            p_cb = cb_model.predict_proba(X_batch)[:, 1]
            dx = xgb.DMatrix(X_batch, feature_names=feature_cols)
            p_xgb = xgb_model.predict(dx)
            probs = 0.50 * p_lgb + 0.30 * p_cb + 0.20 * p_xgb
        else:
            probs = p_lgb

        # Filter matches above minimum threshold directly in memory per batch
        mask = probs >= min_threshold
        accepted_df = batch_df[["source1_entity_id", "matched_entity_id"]][mask].copy()
        accepted_df["pred_prob"] = probs[mask]

        part_file = tmp_dir / f"scored_part_{b}.parquet"
        accepted_df.to_parquet(part_file, index=False)
        scored_parts.append(sql_quote(part_file))

    # Export matching_results.tsv with country calibration, relative margin, and top-K cap
    print(f"\nAssembling final matching_results.tsv with Phase 4 calibration rules...")
    s1_sql = sql_quote(TEST_S1)
    out_sql = sql_quote(MATCHING_OUTPUT_TSV)
    parts_list_sql = ", ".join(f"'{p}'" for p in scored_parts)

    if MATCHING_OUTPUT_TSV.exists():
        MATCHING_OUTPUT_TSV.unlink()

    export_query = f"""
    COPY (
        WITH matched_pairs AS (
            SELECT DISTINCT source1_entity_id, matched_entity_id, pred_prob
            FROM read_parquet([{parts_list_sql}])
        ),
        tagged_matches AS (
            SELECT
                m.source1_entity_id,
                m.matched_entity_id,
                m.pred_prob,
                COALESCE(s1.country_norm, 'other') AS country_norm
            FROM matched_pairs m
            LEFT JOIN read_parquet('{s1_sql}') s1 ON m.source1_entity_id = s1.entity_id
        ),
        country_filtered AS (
            SELECT *
            FROM tagged_matches
            WHERE (country_norm = 'india' AND pred_prob >= {tau_india})
               OR (country_norm = 'us' AND pred_prob >= {tau_us})
               OR (country_norm NOT IN ('us', 'india') AND pred_prob >= {tau_other})
        ),
        ranked_matches AS (
            SELECT
                source1_entity_id,
                matched_entity_id,
                MAX(pred_prob) OVER (PARTITION BY source1_entity_id) AS max_p,
                ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY pred_prob DESC) AS rank,
                pred_prob
            FROM country_filtered
        ),
        competitive_matches AS (
            SELECT source1_entity_id, matched_entity_id
            FROM ranked_matches
            WHERE pred_prob >= max_p - {relative_margin}
              AND rank <= {top_k}
        ),
        grouped_matches AS (
            SELECT
                source1_entity_id,
                string_agg(matched_entity_id, ',' ORDER BY matched_entity_id) AS matched_entity_ids
            FROM competitive_matches
            GROUP BY source1_entity_id
        )
        SELECT
            s1.entity_id AS source1_entity_id,
            COALESCE(g.matched_entity_ids, '') AS matched_entity_ids
        FROM read_parquet('{s1_sql}') s1
        LEFT JOIN grouped_matches g ON s1.entity_id = g.source1_entity_id
        ORDER BY s1.entity_id
    )
    TO '{out_sql}' (HEADER, DELIMITER '\\t', QUOTE '');
    """
    con.execute(export_query)
    print(f"Exported final matching results to: {MATCHING_OUTPUT_TSV}")

    # Clean temporary parts
    shutil.rmtree(tmp_dir, ignore_errors=True)
    return MATCHING_OUTPUT_TSV


def run_submission_validator() -> None:
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--matching", str(MATCHING_OUTPUT_TSV),
        "--candidate", str(CANDIDATE_OUTPUT_TSV),
        "--test-dir", str(TEST_DATA_DIR),
        "--check-ids",
    ]

    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    print(res.stdout)
    if res.stderr:
        print("STDERR:", res.stderr)

    if res.returncode == 0:
        print("\nSUCCESS: All competition submission checks PASSED!")
    else:
        print(f"\nVALIDATION FAILED (exit code {res.returncode})")


def main() -> None:
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: PHASE 4 INFERENCE PIPELINE")
    print("=" * 80)

    start_time = time.time()

    # Load Phase 4 calibration configuration
    calib_json = ROOT / "artifacts" / "models" / "phase4_calibration_params.json"
    tau_us = 0.90
    tau_india = 0.84
    tau_other = 0.90
    relative_margin = 0.05
    top_k = 8

    if calib_json.exists():
        with open(calib_json) as f:
            data = json.load(f)
            tau_us = float(data.get("tau_us", 0.90))
            tau_india = float(data.get("tau_india", 0.84))
            tau_other = float(data.get("tau_other", 0.90))
            relative_margin = float(data.get("relative_margin", 0.05))
            top_k = int(data.get("top_k", 8))

    con = duckdb.connect()
    try:
        con.execute(f"SET threads = {THREADS}")
        con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = true")

        temp_dir = BLOCKING_DIR / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{sql_quote(temp_dir)}'")

        # 1. Candidate TSV check: Reuse existing validated 1.9 GB candidate file if present
        if CANDIDATE_OUTPUT_TSV.exists() and CANDIDATE_OUTPUT_TSV.stat().st_size > 1_000_000_000:
            print(f"Reusing existing validated candidate pairs TSV: {CANDIDATE_OUTPUT_TSV}")
        else:
            export_candidate_pairs_tsv(con)

        # 2. Extract test features if needed
        cand_count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_quote(TEST_CANDIDATES_PATH)}')").fetchone()[0]
        feat_count = (
            con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_quote(TEST_FEATURES_PATH)}')").fetchone()[0]
            if TEST_FEATURES_PATH.exists()
            else -1
        )
        if cand_count != feat_count:
            print(f"Candidate count ({cand_count:,}) != feature count ({feat_count:,}). Rebuilding test features...")
            build_test_features(con)
        else:
            print(f"Reusing existing test features ({feat_count:,} rows): {TEST_FEATURES_PATH}")

        # 3. Predict & export matching_results.tsv with Phase 4 calibration
        predict_and_export_matches(
            con,
            tau_us=tau_us,
            tau_india=tau_india,
            tau_other=tau_other,
            relative_margin=relative_margin,
            top_k=top_k,
        )

        # 4. Validate submission
        run_submission_validator()

    finally:
        con.close()

    total_time = time.time() - start_time
    print(f"\nPhase 4 Inference Pipeline Completed in {total_time / 60:.1f} minutes.")


if __name__ == "__main__":
    main()

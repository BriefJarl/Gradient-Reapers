from __future__ import annotations

"""
End-to-End Test Set Inference & Submission Generation.
Amazon ML Challenge 2026: Business Entity Resolution.

Pipeline:
1. Load normalized test datasets: test_s1, test_s2, test_s3.
2. Build test rare token & exact blocking indexes.
3. Multi-pass candidate generation & optimization for test set.
4. Export output/candidate_pairs.tsv (all test S1 entities included).
5. Compute vectorized features on test candidate pairs using pair_features.py.
6. Score candidate pairs with trained LightGBM model.
7. Apply optimal decision threshold (tau = 0.55).
8. Export output/matching_results.tsv (all test S1 entities included).
9. Run validate_submission.py to verify format compliance.
"""

import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import lightgbm as lgb
import numpy as np
import pandas as pd

from src.features.pair_features import build_feature_select, feature_column_names
from src.models.train_ranker import get_training_feature_names


# ============================================================
# PATHS
# ============================================================

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
MODEL_DIR = ROOT / "artifacts" / "models"
OUTPUT_DIR = ROOT / "output"
TEST_DATA_DIR = ROOT / "student_resource" / "dataset" / "test"
VALIDATE_SCRIPT = ROOT / "student_resource" / "utils" / "validate_submission.py"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_S1 = NORMALIZED_DIR / "test_s1.parquet"
TEST_S2 = NORMALIZED_DIR / "test_s2.parquet"
TEST_S3 = NORMALIZED_DIR / "test_s3.parquet"

MODEL_PATH = MODEL_DIR / "lightgbm_model.txt"
THRESHOLD_JSON = MODEL_DIR / "optimal_threshold.json"

EXPANDED_TEST_CANDIDATES = BLOCKING_DIR / "union" / "test_expanded_candidates.parquet"
TEST_CANDIDATES_PATH = (
    EXPANDED_TEST_CANDIDATES
    if EXPANDED_TEST_CANDIDATES.exists()
    else (BLOCKING_DIR / "union" / "test_optimized_candidates.parquet")
)
TEST_FEATURES_PATH = ROOT / "artifacts" / "features" / "test_features.parquet"

MATCHING_OUTPUT_TSV = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_OUTPUT_TSV = OUTPUT_DIR / "candidate_pairs.tsv"

THREADS = 8
MEMORY_LIMIT = "8GB"
ROW_GROUP_SIZE = 250_000
RARE_TOKEN_TOP_K = 2
MAX_TOKEN_FREQ = 50


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def get_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(f"DESCRIBE SELECT * FROM read_parquet('{sql_quote(path)}')").fetchall()
    return {row[0] for row in rows}


# ============================================================
# STEP 1: TEST RARE TOKEN INDEXES & CANDIDATE GENERATION
# ============================================================

def build_test_rare_indexes(con: duckdb.DuckDBPyConnection) -> dict[str, Path]:
    """Build rare address token indexes for test_s2 and test_s3."""
    index_dir = BLOCKING_DIR / "indexes"
    index_dir.mkdir(parents=True, exist_ok=True)

    test_targets = {
        "S2": TEST_S2,
        "S3": TEST_S3,
    }
    index_paths = {}

    for source_name, source_path in test_targets.items():
        out = index_dir / f"test_{source_name.lower()}_rare_address_token_index.parquet"
        out_sql = sql_quote(out)
        source_sql = sql_quote(source_path)

        print(f"Building rare address index for {source_name}...")
        query = f"""
        COPY (
            WITH exploded_tokens AS (
                SELECT DISTINCT
                    CAST(entity_id AS VARCHAR) AS entity_id,
                    country_norm,
                    LOWER(TRIM(token)) AS token
                FROM read_parquet('{source_sql}')
                CROSS JOIN UNNEST(address_tokens) AS u(token)
                WHERE country_norm IS NOT NULL
                  AND TRIM(country_norm) <> ''
                  AND token IS NOT NULL
                  AND TRIM(token) <> ''
                  AND LENGTH(TRIM(token)) >= 1
            ),
            token_frequency AS (
                SELECT country_norm, token, COUNT(*) AS token_freq
                FROM exploded_tokens
                GROUP BY country_norm, token
                HAVING COUNT(*) <= {MAX_TOKEN_FREQ}
            )
            SELECT e.entity_id, e.country_norm, e.token, f.token_freq
            FROM exploded_tokens AS e
            INNER JOIN token_frequency AS f
                ON e.country_norm = f.country_norm AND e.token = f.token
        )
        TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        con.execute(query)
        index_paths[source_name] = out

    return index_paths


def generate_test_candidates(
    con: duckdb.DuckDBPyConnection,
    rare_index_paths: dict[str, Path],
) -> Path:
    """Generate and optimize test candidate pairs (exact name, address, compact, rare address)."""
    cand_dir = BLOCKING_DIR / "candidates"
    cand_dir.mkdir(parents=True, exist_ok=True)
    part_dir = BLOCKING_DIR / "union" / "test_optimization_parts"
    part_dir.mkdir(parents=True, exist_ok=True)

    s1_sql = sql_quote(TEST_S1)
    s2_sql = sql_quote(TEST_S2)
    s3_sql = sql_quote(TEST_S3)

    print("\n" + "=" * 80)
    print("GENERATING TEST CANDIDATE PAIRS")
    print("=" * 80)

    # 1. Exact Name
    print("Generating test exact name candidates...")
    test_name_out = cand_dir / "test_name_candidates.parquet"
    con.execute(f"""
    COPY (
        SELECT s1.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id, 'S2' AS matched_source, 4 AS block_bit
        FROM read_parquet('{s1_sql}') s1
        INNER JOIN read_parquet('{s2_sql}') t
            ON s1.country_norm = t.country_norm AND s1.name_norm = t.name_norm AND s1.name_norm <> ''
        UNION ALL
        SELECT s1.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id, 'S3' AS matched_source, 4 AS block_bit
        FROM read_parquet('{s1_sql}') s1
        INNER JOIN read_parquet('{s3_sql}') t
            ON s1.country_norm = t.country_norm AND s1.name_norm = t.name_norm AND s1.name_norm <> ''
    ) TO '{sql_quote(test_name_out)}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    # 2. Exact Address
    print("Generating test exact address candidates...")
    test_addr_out = cand_dir / "test_address_candidates.parquet"
    con.execute(f"""
    COPY (
        SELECT s1.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id, 'S2' AS matched_source, 1 AS block_bit
        FROM read_parquet('{s1_sql}') s1
        INNER JOIN read_parquet('{s2_sql}') t
            ON s1.country_norm = t.country_norm AND s1.address_norm = t.address_norm AND s1.address_norm <> ''
        UNION ALL
        SELECT s1.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id, 'S3' AS matched_source, 1 AS block_bit
        FROM read_parquet('{s1_sql}') s1
        INNER JOIN read_parquet('{s3_sql}') t
            ON s1.country_norm = t.country_norm AND s1.address_norm = t.address_norm AND s1.address_norm <> ''
    ) TO '{sql_quote(test_addr_out)}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    # 3. Compact Address
    print("Generating test compact address candidates...")
    test_addrc_out = cand_dir / "test_address_compact_candidates.parquet"
    con.execute(f"""
    COPY (
        SELECT s1.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id, 'S2' AS matched_source, 2 AS block_bit
        FROM read_parquet('{s1_sql}') s1
        INNER JOIN read_parquet('{s2_sql}') t
            ON s1.country_norm = t.country_norm AND s1.address_compact = t.address_compact AND s1.address_compact <> ''
        UNION ALL
        SELECT s1.entity_id AS source1_entity_id, t.entity_id AS matched_entity_id, 'S3' AS matched_source, 2 AS block_bit
        FROM read_parquet('{s1_sql}') s1
        INNER JOIN read_parquet('{s3_sql}') t
            ON s1.country_norm = t.country_norm AND s1.address_compact = t.address_compact AND s1.address_compact <> ''
    ) TO '{sql_quote(test_addrc_out)}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    # 4. Rare Address Tokens
    print("Generating test rare address candidates...")
    test_rare_out = cand_dir / "test_rare_address_candidates.parquet"
    s2_rare_idx = sql_quote(rare_index_paths["S2"])
    s3_rare_idx = sql_quote(rare_index_paths["S3"])

    con.execute(f"""
    COPY (
        WITH s1_tokens AS (
            SELECT DISTINCT s1.entity_id AS source1_entity_id, s1.country_norm, LOWER(TRIM(token)) AS token
            FROM read_parquet('{s1_sql}') s1
            CROSS JOIN UNNEST(s1.address_tokens) AS u(token)
            WHERE s1.country_norm <> '' AND token <> ''
        ),
        s2_match AS (
            SELECT DISTINCT s.source1_entity_id, s.country_norm, s.token, i.token_freq
            FROM s1_tokens s
            INNER JOIN read_parquet('{s2_rare_idx}') i ON s.country_norm = i.country_norm AND s.token = i.token
        ),
        s2_top AS (
            SELECT source1_entity_id, country_norm, token
            FROM s2_match
            QUALIFY ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY token_freq ASC, LENGTH(token) DESC) <= {RARE_TOKEN_TOP_K}
        ),
        s3_match AS (
            SELECT DISTINCT s.source1_entity_id, s.country_norm, s.token, i.token_freq
            FROM s1_tokens s
            INNER JOIN read_parquet('{s3_rare_idx}') i ON s.country_norm = i.country_norm AND s.token = i.token
        ),
        s3_top AS (
            SELECT source1_entity_id, country_norm, token
            FROM s3_match
            QUALIFY ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY token_freq ASC, LENGTH(token) DESC) <= {RARE_TOKEN_TOP_K}
        )
        SELECT DISTINCT s.source1_entity_id, i.entity_id AS matched_entity_id, 'S2' AS matched_source, 16 AS block_bit
        FROM s2_top s
        INNER JOIN read_parquet('{s2_rare_idx}') i ON s.country_norm = i.country_norm AND s.token = i.token
        UNION ALL
        SELECT DISTINCT s.source1_entity_id, i.entity_id AS matched_entity_id, 'S3' AS matched_source, 16 AS block_bit
        FROM s3_top s
        INNER JOIN read_parquet('{s3_rare_idx}') i ON s.country_norm = i.country_norm AND s.token = i.token
    ) TO '{sql_quote(test_rare_out)}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    # 5. Union & Deduplicate
    print("Unioning test candidates...")
    test_union_out = BLOCKING_DIR / "union" / "test_exact_rare_address_union.parquet"
    con.execute(f"""
    COPY (
        WITH raw_cand AS (
            SELECT source1_entity_id, matched_entity_id, matched_source, block_bit FROM read_parquet('{sql_quote(test_name_out)}')
            UNION ALL
            SELECT source1_entity_id, matched_entity_id, matched_source, block_bit FROM read_parquet('{sql_quote(test_addr_out)}')
            UNION ALL
            SELECT source1_entity_id, matched_entity_id, matched_source, block_bit FROM read_parquet('{sql_quote(test_addrc_out)}')
            UNION ALL
            SELECT source1_entity_id, matched_entity_id, matched_source, block_bit FROM read_parquet('{sql_quote(test_rare_out)}')
        ),
        grouped AS (
            SELECT source1_entity_id, matched_entity_id, matched_source, bit_or(block_bit) AS blocking_mask
            FROM raw_cand
            GROUP BY source1_entity_id, matched_entity_id, matched_source
        )
        SELECT
            source1_entity_id,
            matched_entity_id,
            matched_source,
            blocking_mask,
            bit_count(blocking_mask) AS num_blocking_methods,
            concat_ws('|',
                CASE WHEN (blocking_mask & 1) <> 0 THEN 'address' END,
                CASE WHEN (blocking_mask & 2) <> 0 THEN 'address_compact' END,
                CASE WHEN (blocking_mask & 4) <> 0 THEN 'name' END,
                CASE WHEN (blocking_mask & 16) <> 0 THEN 'rare_address' END
            ) AS blocking_methods
        FROM grouped
    ) TO '{sql_quote(test_union_out)}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """)

    # 6. Optimize & Prune
    print("Optimizing test candidates...")
    con.execute(f"""
    COPY (
        WITH base AS (
            SELECT * FROM read_parquet('{sql_quote(test_union_out)}')
        ),
        multi_part AS (
            SELECT * FROM base WHERE num_blocking_methods >= 2
        ),
        name_s2 AS (
            SELECT c.* FROM base c
            INNER JOIN read_parquet('{s1_sql}') s1 ON c.source1_entity_id = s1.entity_id
            INNER JOIN read_parquet('{s2_sql}') s2 ON c.matched_entity_id = s2.entity_id
            WHERE c.num_blocking_methods = 1 AND c.blocking_methods = 'name' AND c.matched_source = 'S2'
              AND (s1.name_norm = s2.name_norm OR s1.name_compact = s2.name_compact)
        ),
        name_s3 AS (
            SELECT c.* FROM base c
            INNER JOIN read_parquet('{s1_sql}') s1 ON c.source1_entity_id = s1.entity_id
            INNER JOIN read_parquet('{s3_sql}') s3 ON c.matched_entity_id = s3.entity_id
            WHERE c.num_blocking_methods = 1 AND c.blocking_methods = 'name' AND c.matched_source = 'S3'
              AND (s1.name_norm = s3.name_norm OR s1.name_compact = s3.name_compact)
        ),
        rare_s2 AS (
            SELECT c.* FROM base c
            INNER JOIN read_parquet('{s1_sql}') s1 ON c.source1_entity_id = s1.entity_id
            INNER JOIN read_parquet('{s2_sql}') s2 ON c.matched_entity_id = s2.entity_id
            WHERE c.num_blocking_methods = 1 AND c.blocking_methods = 'rare_address' AND c.matched_source = 'S2'
              AND (s1.name_norm = s2.name_norm OR s1.name_compact = s2.name_compact
                   OR COALESCE(len(list_intersect(s1.address_tokens, s2.address_tokens)) >= 3, FALSE))
        ),
        rare_s3 AS (
            SELECT c.* FROM base c
            INNER JOIN read_parquet('{s1_sql}') s1 ON c.source1_entity_id = s1.entity_id
            INNER JOIN read_parquet('{s3_sql}') s3 ON c.matched_entity_id = s3.entity_id
            WHERE c.num_blocking_methods = 1 AND c.blocking_methods = 'rare_address' AND c.matched_source = 'S3'
              AND (s1.name_norm = s3.name_norm OR s1.name_compact = s3.name_compact
                   OR COALESCE(len(list_intersect(s1.address_tokens, s3.address_tokens)) >= 3, FALSE))
        )
        SELECT * FROM multi_part
        UNION ALL SELECT * FROM name_s2
        UNION ALL SELECT * FROM name_s3
        UNION ALL SELECT * FROM rare_s2
        UNION ALL SELECT * FROM rare_s3
    ) TO '{sql_quote(TEST_CANDIDATES_PATH)}' (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE 500000);
    """)

    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{sql_quote(TEST_CANDIDATES_PATH)}')").fetchone()[0]
    print(f"Total optimized test candidates: {count:,}")
    return TEST_CANDIDATES_PATH


# ============================================================
# STEP 2: WRITE candidate_pairs.tsv
# ============================================================

def export_candidate_pairs_tsv(con: duckdb.DuckDBPyConnection, num_parts: int = 4) -> None:
    """Export output/candidate_pairs.tsv ensuring every test S1 entity is present exactly once."""
    print("\n" + "=" * 80)
    print("EXPORTING candidate_pairs.tsv (Partitioned)")
    print("=" * 80)

    s1_sql = sql_quote(TEST_S1)
    cand_sql = sql_quote(TEST_CANDIDATES_PATH)
    out_path = CANDIDATE_OUTPUT_TSV

    if out_path.exists():
        out_path.unlink()

    tmp_dir = BLOCKING_DIR / "tmp" / "candidate_tsv_parts"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    part_files = []

    for p in range(num_parts):
        print(f"  Exporting candidate partition {p+1}/{num_parts}...")
        part_tsv = tmp_dir / f"cand_part_{p}.tsv"
        if part_tsv.exists():
            part_tsv.unlink()

        query = f"""
        COPY (
            WITH s1_part AS (
                SELECT entity_id AS source1_entity_id
                FROM read_parquet('{s1_sql}')
                WHERE MOD(ABS(HASH(entity_id)), {num_parts}) = {p}
            ),
            grouped_cands AS (
                SELECT
                    source1_entity_id,
                    string_agg(matched_entity_id, ',') AS candidate_entity_ids
                FROM read_parquet('{cand_sql}')
                WHERE MOD(ABS(HASH(source1_entity_id)), {num_parts}) = {p}
                GROUP BY source1_entity_id
            )
            SELECT
                s.source1_entity_id,
                COALESCE(g.candidate_entity_ids, '') AS candidate_entity_ids
            FROM s1_part s
            LEFT JOIN grouped_cands g ON s.source1_entity_id = g.source1_entity_id
            ORDER BY s.source1_entity_id
        )
        TO '{sql_quote(part_tsv)}' (HEADER FALSE, DELIMITER '\\t', QUOTE '');
        """
        con.execute(query)
        part_files.append(part_tsv)

    print("Concatenating candidate partitions to candidate_pairs.tsv...")
    with open(out_path, "w", encoding="utf-8") as out_f:
        out_f.write("source1_entity_id\tcandidate_entity_ids\n")
        for part_tsv in part_files:
            with open(part_tsv, "r", encoding="utf-8") as in_f:
                shutil.copyfileobj(in_f, out_f)
            part_tsv.unlink()

    print(f"Exported candidate pairs to: {out_path}")


# ============================================================
# STEP 3: EXTRACT TEST FEATURES & PREDICT
# ============================================================

def build_test_features(con: duckdb.DuckDBPyConnection) -> Path:
    """Compute vectorized similarity features on test candidates."""
    print("\n" + "=" * 80)
    print("COMPUTING TEST CANDIDATE FEATURES")
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

    s2_out = TEST_FEATURES_PATH.parent / "test_features_s2.parquet"
    s3_out = TEST_FEATURES_PATH.parent / "test_features_s3.parquet"
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
    threshold: float,
) -> Path:
    """Score test features with LightGBM and write output/matching_results.tsv."""
    print("\n" + "=" * 80)
    print(f"SCORING TEST CANDIDATES (Threshold = {threshold:.2f})")
    print("=" * 80)

    feature_cols = get_training_feature_names()
    feat_sql = sql_quote(TEST_FEATURES_PATH)
    feature_list_sql = ", ".join(f'"{c}"' for c in feature_cols)

    # 1. Load model
    print(f"Loading model booster from: {MODEL_PATH}")
    booster = lgb.Booster(model_file=str(MODEL_PATH))

    # 2. Score in batches to stay within RAM limits
    print("Scoring test candidate feature batches...")
    batch_size = 1_000_000
    total_rows = con.execute(f"SELECT COUNT(*) FROM read_parquet('{feat_sql}')").fetchone()[0]
    num_batches = (total_rows + batch_size - 1) // batch_size

    scored_parts = []
    tmp_dir = BLOCKING_DIR / "tmp" / "test_scored_parts"
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
        probs = booster.predict(X_batch)

        # Filter matches above threshold directly in memory per batch
        mask = probs >= threshold
        accepted_df = batch_df[["source1_entity_id", "matched_entity_id"]][mask]

        part_file = tmp_dir / f"scored_part_{b}.parquet"
        accepted_df.to_parquet(part_file, index=False)
        scored_parts.append(sql_quote(part_file))

    # 3. Export matching_results.tsv using DuckDB
    print("\nAssembling final matching_results.tsv...")
    s1_sql = sql_quote(TEST_S1)
    out_sql = sql_quote(MATCHING_OUTPUT_TSV)
    parts_list_sql = ", ".join(f"'{p}'" for p in scored_parts)

    if MATCHING_OUTPUT_TSV.exists():
        MATCHING_OUTPUT_TSV.unlink()

    export_query = f"""
    COPY (
        WITH matched_pairs AS (
            SELECT DISTINCT source1_entity_id, matched_entity_id
            FROM read_parquet([{parts_list_sql}])
        ),
        grouped_matches AS (
            SELECT
                source1_entity_id,
                string_agg(matched_entity_id, ',' ORDER BY matched_entity_id) AS matched_entity_ids
            FROM matched_pairs
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

    return MATCHING_OUTPUT_TSV


# ============================================================
# STEP 4: SUBMISSION VALIDATION
# ============================================================

def run_submission_validator() -> None:
    """Run the official competition validator on the output files."""
    print("\n" + "=" * 80)
    print("RUNNING OFFICIAL SUBMISSION VALIDATOR")
    print("=" * 80)

    cmd = [
        sys.executable,
        str(VALIDATE_SCRIPT),
        "--matching", str(MATCHING_OUTPUT_TSV),
        "--candidate", str(CANDIDATE_OUTPUT_TSV),
        "--test-dir", str(TEST_DATA_DIR),
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


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026: TEST INFERENCE PIPELINE")
    print("=" * 80)

    start_time = time.time()

    # Load optimal threshold
    threshold = 0.55
    if THRESHOLD_JSON.exists():
        with open(THRESHOLD_JSON) as f:
            data = json.load(f)
            threshold = float(data.get("best_threshold", 0.55))
    print(f"Using Decision Threshold: {threshold:.2f}\n")

    con = duckdb.connect()

    try:
        con.execute(f"SET threads = {THREADS}")
        con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = true")

        temp_dir = BLOCKING_DIR / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory = '{sql_quote(temp_dir)}'")

        # 1. Rare indexes for test
        if not TEST_CANDIDATES_PATH.exists():
            rare_indexes = build_test_rare_indexes(con)
            generate_test_candidates(con, rare_indexes)
        else:
            print(f"Using existing test candidates: {TEST_CANDIDATES_PATH}")

        # 2. Export candidate_pairs.tsv
        export_candidate_pairs_tsv(con)

        # 3. Extract features
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
            print(f"Using existing test features ({feat_count:,} rows): {TEST_FEATURES_PATH}")

        # 4. Predict & export matching_results.tsv
        predict_and_export_matches(con, threshold)

        # 5. Validate submission
        run_submission_validator()

    finally:
        con.close()

    total_time = time.time() - start_time
    print(f"\nInference Pipeline Completed in {total_time / 60:.1f} minutes.")


if __name__ == "__main__":
    main()

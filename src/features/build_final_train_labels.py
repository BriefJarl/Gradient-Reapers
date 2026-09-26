from __future__ import annotations

"""
Build labels for the FINAL Amazon ML Challenge 2026 candidate pool.

Input:
    artifacts/features/final_train/train_features.parquet

Ground truth:
    artifacts/blocking/ground_truth_pairs.parquet

Output:
    artifacts/features/final_train/train_features_labeled.parquet

Design:
- DuckDB only; no pandas.
- Labels are pair-level: 1 if candidate pair exists in ground truth, else 0.
- Candidate identity is (source1_entity_id, matched_entity_id, matched_source).
- Validates uniqueness before and after the join.
- Uses a partial file and renames only after validation.
"""

import os
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "artifacts" / "features" / "final_train"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
TMP_DIR = BLOCKING_DIR / "duckdb_tmp"

FEATURES = FEATURE_DIR / "train_features.parquet"
GROUND_TRUTH = BLOCKING_DIR / "ground_truth_pairs.parquet"
OUTPUT = FEATURE_DIR / "train_features_labeled.parquet"
PARTIAL = FEATURE_DIR / "train_features_labeled.partial.parquet"

EXPECTED_CANDIDATES = 41_982_254
EXPECTED_TRUE_PAIRS = 7_638_365

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY", "8GB")
THREADS = int(
    os.environ.get(
        "DUCKDB_THREADS",
        str(max(4, min(12, (os.cpu_count() or 10) - 2))),
    )
)


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def configure(con: duckdb.DuckDBPyConnection) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{sql_quote(TMP_DIR)}'")


def count_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_quote(path)}')"
        ).fetchone()[0]
    )


def main() -> None:
    print("=" * 88)
    print("AMAZON ML CHALLENGE 2026")
    print("FINAL TRAIN LABEL BUILDER")
    print("=" * 88)

    for path in (FEATURES, GROUND_TRUTH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found:\n{path}")

    con = duckdb.connect()

    try:
        configure(con)

        print()
        print(f"FEATURES     : {FEATURES}")
        print(f"GROUND TRUTH : {GROUND_TRUTH}")
        print(f"OUTPUT       : {OUTPUT}")
        print(f"DuckDB       : {THREADS} threads / {MEMORY_LIMIT}")

        # ------------------------------------------------------------
        # 1. Input counts
        # ------------------------------------------------------------
        candidate_count = count_rows(con, FEATURES)
        gt_count = count_rows(con, GROUND_TRUTH)

        print()
        print("[1/5] INPUT COUNTS")
        print(f"Candidate rows : {candidate_count:,}")
        print(f"GT true pairs  : {gt_count:,}")

        if candidate_count != EXPECTED_CANDIDATES:
            raise RuntimeError(
                f"Unexpected candidate count: {candidate_count:,}; "
                f"expected {EXPECTED_CANDIDATES:,}"
            )

        if gt_count != EXPECTED_TRUE_PAIRS:
            raise RuntimeError(
                f"Unexpected ground-truth pair count: {gt_count:,}; "
                f"expected {EXPECTED_TRUE_PAIRS:,}"
            )

        # ------------------------------------------------------------
        # 2. Validate GT uniqueness
        # ------------------------------------------------------------
        print()
        print("[2/5] VALIDATING GROUND TRUTH")

        gt_duplicates = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT
                        source1_entity_id,
                        matched_entity_id,
                        matched_source
                    FROM read_parquet('{sql_quote(GROUND_TRUTH)}')
                    GROUP BY 1, 2, 3
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )

        if gt_duplicates:
            raise RuntimeError(
                f"Ground truth contains {gt_duplicates:,} duplicate pair groups."
            )

        invalid_sources = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_quote(GROUND_TRUTH)}')
                WHERE matched_source NOT IN ('S2', 'S3')
                """
            ).fetchone()[0]
        )

        if invalid_sources:
            raise RuntimeError(
                f"Ground truth contains {invalid_sources:,} invalid source labels."
            )

        print("Ground-truth duplicate pairs : 0")
        print("Ground-truth source labels    : PASS")

        # ------------------------------------------------------------
        # 3. Build labels
        #
        # LEFT JOIN preserves every candidate, including negatives.
        # GT is unique, so the join cannot multiply candidate rows.
        # ------------------------------------------------------------
        print()
        print("[3/5] BUILDING FINAL LABELS")

        if PARTIAL.exists():
            PARTIAL.unlink()

        con.execute(
            f"""
            COPY
            (
                SELECT
                    f.*,

                    CAST(
                        CASE
                            WHEN g.source1_entity_id IS NOT NULL
                            THEN 1
                            ELSE 0
                        END
                        AS TINYINT
                    ) AS label

                FROM read_parquet('{sql_quote(FEATURES)}') f

                LEFT JOIN read_parquet('{sql_quote(GROUND_TRUTH)}') g
                  ON f.source1_entity_id = g.source1_entity_id
                 AND f.matched_entity_id = g.matched_entity_id
                 AND f.matched_source = g.matched_source
            )
            TO '{sql_quote(PARTIAL)}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        # ------------------------------------------------------------
        # 4. Validate output
        # ------------------------------------------------------------
        print()
        print("[4/5] VALIDATING LABELED DATA")

        labeled_count = count_rows(con, PARTIAL)

        duplicate_groups = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT
                        source1_entity_id,
                        matched_entity_id,
                        matched_source
                    FROM read_parquet('{sql_quote(PARTIAL)}')
                    GROUP BY 1, 2, 3
                    HAVING COUNT(*) > 1
                )
                """
            ).fetchone()[0]
        )

        positive_count = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_quote(PARTIAL)}')
                WHERE label = 1
                """
            ).fetchone()[0]
        )

        negative_count = labeled_count - positive_count

        bad_labels = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_quote(PARTIAL)}')
                WHERE label NOT IN (0, 1)
                   OR label IS NULL
                """
            ).fetchone()[0]
        )

        source_rows = con.execute(
            f"""
            SELECT
                matched_source,
                COUNT(*) AS candidates,
                SUM(label) AS positives
            FROM read_parquet('{sql_quote(PARTIAL)}')
            GROUP BY matched_source
            ORDER BY matched_source
            """
        ).fetchall()

        print(f"Labeled rows          : {labeled_count:,}")
        print(f"Positive pairs        : {positive_count:,}")
        print(f"Negative pairs        : {negative_count:,}")
        print(
            f"Positive rate         : "
            f"{positive_count / labeled_count:.6%}"
        )
        print(f"Duplicate pair groups : {duplicate_groups:,}")
        print(f"Invalid labels        : {bad_labels:,}")

        print()
        print("SOURCE-WISE LABELS")
        for source, candidates, positives in source_rows:
            print(
                f"{source}: "
                f"{candidates:,} candidates, "
                f"{positives:,} positives, "
                f"{positives / candidates:.6%}"
            )

        if labeled_count != EXPECTED_CANDIDATES:
            raise RuntimeError(
                f"Labeled row count mismatch: "
                f"{labeled_count:,} != {EXPECTED_CANDIDATES:,}"
            )

        if positive_count > EXPECTED_TRUE_PAIRS:
            raise RuntimeError(
                "Candidate positives cannot exceed total ground-truth pairs."
            )

        if duplicate_groups != 0:
            raise RuntimeError(
                f"Duplicate labeled pairs detected: {duplicate_groups:,}"
            )

        if bad_labels != 0:
            raise RuntimeError(
                f"Invalid labels detected: {bad_labels:,}"
            )

        # ------------------------------------------------------------
        # 5. Atomic-ish finalization
        # ------------------------------------------------------------
        print()
        print("[5/5] FINALIZING OUTPUT")

        if OUTPUT.exists():
            OUTPUT.unlink()

        PARTIAL.replace(OUTPUT)

        print()
        print("=" * 88)
        print("FINAL TRAIN LABEL BUILD PASSED")
        print("=" * 88)
        print(f"Output     : {OUTPUT}")
        print(f"Rows       : {labeled_count:,}")
        print(f"Positives  : {positive_count:,}")
        print(f"Negatives  : {negative_count:,}")
        print("=" * 88)

    finally:
        con.close()


if __name__ == "__main__":
    main()

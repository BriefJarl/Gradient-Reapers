from __future__ import annotations

"""
Create a deterministic S1-level train/validation split for the FINAL
Amazon ML Challenge 2026 labeled candidate pool.

All candidate pairs belonging to the same Source-1 entity are placed in
the same partition. This prevents pair-level leakage.

Input:
    artifacts/features/final_train/train_features_labeled.parquet

Outputs:
    artifacts/features/final_train/train_split.parquet
    artifacts/features/final_train/valid_split.parquet
"""

import os
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "artifacts" / "features" / "final_train"
TMP_DIR = ROOT / "artifacts" / "blocking" / "duckdb_tmp"

INPUT = FEATURE_DIR / "train_features_labeled.parquet"
TRAIN_OUTPUT = FEATURE_DIR / "train_split.parquet"
VALID_OUTPUT = FEATURE_DIR / "valid_split.parquet"

EXPECTED_TOTAL = 41_982_254

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
    print("FINAL S1-LEVEL TRAIN / VALIDATION SPLIT")
    print("=" * 88)

    if not INPUT.exists():
        raise FileNotFoundError(f"Required file not found:\n{INPUT}")

    con = duckdb.connect()

    try:
        configure(con)

        total = count_rows(con, INPUT)

        print()
        print(f"INPUT      : {INPUT}")
        print(f"TRAIN      : {TRAIN_OUTPUT}")
        print(f"VALID      : {VALID_OUTPUT}")
        print(f"Rows       : {total:,}")
        print(f"DuckDB     : {THREADS} threads / {MEMORY_LIMIT}")

        if total != EXPECTED_TOTAL:
            raise RuntimeError(
                f"Unexpected labeled row count: "
                f"{total:,} != {EXPECTED_TOTAL:,}"
            )

        if TRAIN_OUTPUT.exists():
            TRAIN_OUTPUT.unlink()

        if VALID_OUTPUT.exists():
            VALID_OUTPUT.unlink()

        input_sql = sql_quote(INPUT)

        # ------------------------------------------------------------
        # Use the same deterministic hash rule for every row belonging
        # to an S1 entity. hash() is deterministic within DuckDB.
        #
        # 8 buckets -> TRAIN
        # 2 buckets -> VALID
        # ------------------------------------------------------------

        print()
        print("[1/4] CREATING TRAIN SPLIT")

        con.execute(
            f"""
            COPY
            (
                SELECT *
                FROM read_parquet('{input_sql}')
                WHERE hash(source1_entity_id) % 10 < 8
            )
            TO '{sql_quote(TRAIN_OUTPUT)}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        print()
        print("[2/4] CREATING VALIDATION SPLIT")

        con.execute(
            f"""
            COPY
            (
                SELECT *
                FROM read_parquet('{input_sql}')
                WHERE hash(source1_entity_id) % 10 >= 8
            )
            TO '{sql_quote(VALID_OUTPUT)}'
            (
                FORMAT PARQUET,
                COMPRESSION ZSTD,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        train_count = count_rows(con, TRAIN_OUTPUT)
        valid_count = count_rows(con, VALID_OUTPUT)

        # ------------------------------------------------------------
        # Validate partition completeness.
        # ------------------------------------------------------------
        print()
        print("[3/4] VALIDATING PARTITION")

        overlap = int(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM (
                    SELECT DISTINCT source1_entity_id
                    FROM read_parquet('{sql_quote(TRAIN_OUTPUT)}')
                ) t
                INNER JOIN (
                    SELECT DISTINCT source1_entity_id
                    FROM read_parquet('{sql_quote(VALID_OUTPUT)}')
                ) v
                  ON t.source1_entity_id = v.source1_entity_id
                """
            ).fetchone()[0]
        )

        if train_count + valid_count != total:
            raise RuntimeError(
                f"Partition row mismatch: "
                f"{train_count:,} + {valid_count:,} != {total:,}"
            )

        if overlap != 0:
            raise RuntimeError(
                f"S1 leakage detected: {overlap:,} entities appear in both splits."
            )

        train_s1 = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT source1_entity_id)
                FROM read_parquet('{sql_quote(TRAIN_OUTPUT)}')
                """
            ).fetchone()[0]
        )

        valid_s1 = int(
            con.execute(
                f"""
                SELECT COUNT(DISTINCT source1_entity_id)
                FROM read_parquet('{sql_quote(VALID_OUTPUT)}')
                """
            ).fetchone()[0]
        )

        print(f"Train rows       : {train_count:,}")
        print(f"Validation rows  : {valid_count:,}")
        print(f"Total            : {train_count + valid_count:,}")
        print(f"Train S1 entities: {train_s1:,}")
        print(f"Valid S1 entities: {valid_s1:,}")
        print(f"S1 overlap       : {overlap}")

        print()
        print("[4/4] SPLIT SANITY")

        if overlap != 0:
            raise RuntimeError("S1-level split validation failed.")

        print("S1-level split check: PASS")

        print()
        print("=" * 88)
        print("FINAL TRAIN / VALIDATION SPLIT PASSED")
        print("=" * 88)
        print(f"TRAIN : {TRAIN_OUTPUT}")
        print(f"VALID : {VALID_OUTPUT}")
        print("=" * 88)

    finally:
        con.close()


if __name__ == "__main__":
    main()
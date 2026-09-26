from __future__ import annotations

"""
Production feature builder for the FINAL Amazon ML Challenge 2026 train
candidate pool.

Key properties:
- Uses the already-validated vectorized feature definitions in pair_features.py.
- Reads FINAL candidates only:
    artifacts/blocking/union/train_final_candidates.parquet
- Writes only under:
    artifacts/features/final_train/
- DuckDB-native; no pandas; no Python row loops.
- Atomic-ish per-file workflow: write to *.partial.parquet, validate, then rename.
- Resumable: valid completed S2/S3 files are reused unless --force.
"""

import argparse
import os
from pathlib import Path

import duckdb

try:
    from src.features.pair_features import build_feature_select, feature_column_names
except ModuleNotFoundError:
    from pair_features import build_feature_select, feature_column_names


ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
FEATURE_DIR = ROOT / "artifacts" / "features" / "final_train"
TMP_DIR = BLOCKING_DIR / "duckdb_tmp"

CANDIDATES = BLOCKING_DIR / "union" / "train_final_candidates.parquet"

S1_PATH = NORMALIZED_DIR / "train_s1.parquet"
S2_PATH = NORMALIZED_DIR / "train_s2.parquet"
S3_PATH = NORMALIZED_DIR / "train_s3.parquet"

EXPECTED_ROWS = {
    "S2": 21_053_333,
    "S3": 20_928_921,
}
EXPECTED_TOTAL = 41_982_254

ROW_GROUP_SIZE = 250_000
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


def count_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_quote(path)}')"
        ).fetchone()[0]
    )


def get_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    rows = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{sql_quote(path)}')"
    ).fetchall()
    return {row[0] for row in rows}


def configure(con: duckdb.DuckDBPyConnection) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{sql_quote(TMP_DIR)}'")


def expected_output(source: str) -> Path:
    return FEATURE_DIR / f"train_features_{source.lower()}.parquet"


def partial_output(source: str) -> Path:
    return FEATURE_DIR / f"train_features_{source.lower()}.partial.parquet"


def validate_source_output(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    source: str,
    expected_rows: int,
) -> None:
    if not path.exists():
        raise FileNotFoundError(path)

    columns = get_columns(con, path)
    expected_columns = set(feature_column_names())

    missing = expected_columns - columns
    if missing:
        raise RuntimeError(
            f"{source}: missing feature columns: {sorted(missing)}"
        )

    rows = count_rows(con, path)
    if rows != expected_rows:
        raise RuntimeError(
            f"{source}: row-count mismatch: expected {expected_rows:,}, "
            f"got {rows:,}"
        )

    checks = con.execute(
        f"""
        SELECT
            COUNT(*) FILTER (WHERE source1_entity_id IS NULL) AS s1_nulls,
            COUNT(*) FILTER (WHERE matched_entity_id IS NULL) AS target_nulls,
            COUNT(*) FILTER (WHERE matched_source <> '{source}') AS bad_source,
            COUNT(*) FILTER (
                WHERE name_exact IS NULL
                   OR address_exact IS NULL
                   OR country_exact IS NULL
                   OR name_char_ratio IS NULL
                   OR address_char_ratio IS NULL
            ) AS feature_nulls,
            COUNT(*) FILTER (
                WHERE name_char_ratio < 0 OR name_char_ratio > 1
                   OR address_char_ratio < 0 OR address_char_ratio > 1
                   OR name_token_overlap < 0 OR name_token_overlap > 1
                   OR name_token_jaccard < 0 OR name_token_jaccard > 1
                   OR address_token_overlap < 0 OR address_token_overlap > 1
                   OR address_token_jaccard < 0 OR address_token_jaccard > 1
                   OR name_length_ratio < 0 OR name_length_ratio > 1
                   OR address_length_ratio < 0 OR address_length_ratio > 1
            ) AS bad_ranges
        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchone()

    if any(value != 0 for value in checks):
        raise RuntimeError(
            f"{source}: validation failed: "
            f"s1_nulls={checks[0]}, target_nulls={checks[1]}, "
            f"bad_source={checks[2]}, feature_nulls={checks[3]}, "
            f"bad_ranges={checks[4]}"
        )

    # The candidate pool has already passed pair-identity validation.
    # This additional check catches accidental target-table duplication.
    duplicate_groups = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source
                FROM read_parquet('{sql_quote(path)}')
                GROUP BY 1, 2, 3
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )

    if duplicate_groups:
        raise RuntimeError(
            f"{source}: duplicate feature pairs detected: {duplicate_groups:,}"
        )


def build_source(
    con: duckdb.DuckDBPyConnection,
    source: str,
    *,
    force: bool,
    sample_rows: int,
) -> Path:
    output = expected_output(source)
    partial = partial_output(source)
    expected_rows = EXPECTED_ROWS[source]

    if sample_rows > 0:
        output = FEATURE_DIR / f"train_features_sample_{source.lower()}.parquet"
        partial = FEATURE_DIR / f"train_features_sample_{source.lower()}.partial.parquet"

    if output.exists() and not force:
        print(f"\n[{source}] Existing output found: {output}")
        if sample_rows == 0:
            try:
                validate_source_output(con, output, source, expected_rows)
                print(f"[{source}] VALID EXISTING OUTPUT -> SKIP")
                return output
            except Exception as exc:
                print(f"[{source}] Existing output invalid -> rebuild: {exc}")
        else:
            print(f"[{source}] Sample output exists -> rebuild")

    if partial.exists():
        partial.unlink()

    target_path = S2_PATH if source == "S2" else S3_PATH

    s1_columns = get_columns(con, S1_PATH)
    target_columns = get_columns(con, target_path)

    select_features = build_feature_select(
        s1_alias="s1",
        target_alias="t",
        candidate_alias="c",
        s1_columns=s1_columns,
        target_columns=target_columns,
    )

    limit_clause = f"LIMIT {sample_rows}" if sample_rows > 0 else ""

    print("\n" + "=" * 80)
    print(f"BUILDING FINAL TRAIN FEATURES: {source}")
    print("=" * 80)
    print(f"Candidates : {CANDIDATES}")
    print(f"Target     : {target_path}")
    print(f"Output     : {output}")
    print(f"Expected   : {expected_rows:,}" if sample_rows == 0 else
          f"Sample     : {sample_rows:,}")
    print(f"DuckDB     : {THREADS} threads / {MEMORY_LIMIT}")
    print()

    query = f"""
        COPY
        (
            SELECT
                {select_features}
            FROM
            (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    num_blocking_methods,
                    blocking_methods
                FROM read_parquet('{sql_quote(CANDIDATES)}')
                WHERE matched_source = '{source}'
                {limit_clause}
            ) c
            INNER JOIN read_parquet('{sql_quote(S1_PATH)}') s1
                ON c.source1_entity_id = s1.entity_id
            INNER JOIN read_parquet('{sql_quote(target_path)}') t
                ON c.matched_entity_id = t.entity_id
        )
        TO '{sql_quote(partial)}'
        (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            ROW_GROUP_SIZE {ROW_GROUP_SIZE}
        )
    """

    con.execute(query)

    if sample_rows > 0:
        rows = count_rows(con, partial)
        if rows != sample_rows:
            raise RuntimeError(
                f"{source}: sample row mismatch: expected {sample_rows:,}, "
                f"got {rows:,}"
            )
        partial.replace(output)
        print(f"[{source}] SAMPLE COMPLETE: {rows:,} rows")
        return output

    validate_source_output(con, partial, source, expected_rows)
    partial.replace(output)

    # Validate the final renamed file too.
    validate_source_output(con, output, source, expected_rows)

    print(f"[{source}] COMPLETE: {expected_rows:,} rows")
    return output


def combine_outputs(
    con: duckdb.DuckDBPyConnection,
    s2_output: Path,
    s3_output: Path,
    *,
    force: bool,
) -> Path:
    final_output = FEATURE_DIR / "train_features.parquet"
    partial = FEATURE_DIR / "train_features.partial.parquet"

    if final_output.exists() and not force:
        try:
            rows = count_rows(con, final_output)
            if rows == EXPECTED_TOTAL:
                print(f"\n[FINAL] Existing combined output is valid by row count: {rows:,}")
                return final_output
        except Exception:
            pass

    if partial.exists():
        partial.unlink()

    print("\n" + "=" * 80)
    print("COMBINING FINAL TRAIN FEATURES")
    print("=" * 80)

    con.execute(
        f"""
        COPY
        (
            SELECT * FROM read_parquet('{sql_quote(s2_output)}')
            UNION ALL
            SELECT * FROM read_parquet('{sql_quote(s3_output)}')
        )
        TO '{sql_quote(partial)}'
        (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            ROW_GROUP_SIZE {ROW_GROUP_SIZE}
        )
        """
    )

    rows = count_rows(con, partial)
    if rows != EXPECTED_TOTAL:
        raise RuntimeError(
            f"Final feature row mismatch: expected {EXPECTED_TOTAL:,}, got {rows:,}"
        )

    partial.replace(final_output)

    # Lightweight final integrity checks.
    bad_source_rows = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(final_output)}')
            WHERE matched_source NOT IN ('S2', 'S3')
            """
        ).fetchone()[0]
    )
    if bad_source_rows:
        raise RuntimeError(f"Invalid matched_source rows: {bad_source_rows:,}")

    print(f"Final rows : {rows:,}")
    print(f"Output     : {final_output}")

    return final_output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final Amazon ML Challenge train pair features."
    )
    parser.add_argument(
        "--source",
        choices=["S2", "S3", "all"],
        default="all",
    )
    parser.add_argument(
        "--sample-rows",
        type=int,
        default=0,
        help="Small smoke test per source. 0 = full dataset.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild existing outputs.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("FINAL TRAIN FEATURE ENGINE")
    print("=" * 80)

    required = [CANDIDATES, S1_PATH, S2_PATH, S3_PATH]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    print(f"\nFINAL candidates : {CANDIDATES}")
    print(f"FINAL feature dir: {FEATURE_DIR}")

    con = duckdb.connect()

    try:
        configure(con)

        # Confirm the frozen final candidate pool before doing expensive work.
        candidate_rows = count_rows(con, CANDIDATES)
        if candidate_rows != EXPECTED_TOTAL:
            raise RuntimeError(
                f"Final candidate count mismatch: expected {EXPECTED_TOTAL:,}, "
                f"got {candidate_rows:,}"
            )

        if args.sample_rows > 0:
            build_source(
                con, "S2", force=True, sample_rows=args.sample_rows
            )
            build_source(
                con, "S3", force=True, sample_rows=args.sample_rows
            )
            print("\n" + "=" * 80)
            print("FINAL TRAIN FEATURE SMOKE TEST PASSED")
            print("=" * 80)
            return

        s2_output = build_source(
            con, "S2", force=args.force, sample_rows=0
        )
        s3_output = build_source(
            con, "S3", force=args.force, sample_rows=0
        )

        final_output = combine_outputs(
            con,
            s2_output,
            s3_output,
            force=args.force,
        )

        print("\n" + "=" * 80)
        print("FINAL TRAIN FEATURE BUILD PASSED")
        print("=" * 80)
        print(f"S2       : {s2_output}")
        print(f"S3       : {s3_output}")
        print(f"COMBINED : {final_output}")
        print(f"ROWS     : {count_rows(con, final_output):,}")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()

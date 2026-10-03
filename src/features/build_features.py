from __future__ import annotations

import argparse
from pathlib import Path

import duckdb

try:
    # Works when executed as a Python module:
    # python -m src.features.build_features
    from src.features.pair_features import (
        build_feature_select,
        feature_column_names,
    )
except ModuleNotFoundError:
    # Works when executed directly:
    # python .\src\features\build_features.py
    from pair_features import (
        build_feature_select,
        feature_column_names,
    )




ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
FEATURE_DIR = ROOT / "artifacts" / "features"

PHASE3_CANDIDATES = (
    BLOCKING_DIR
    / "union"
    / "train_phase3_candidates.parquet"
)

EXPANDED_CANDIDATES = (
    BLOCKING_DIR
    / "union"
    / "train_expanded_candidates.parquet"
)

CANDIDATES = (
    PHASE3_CANDIDATES
    if PHASE3_CANDIDATES.exists()
    else (
        EXPANDED_CANDIDATES
        if EXPANDED_CANDIDATES.exists()
        else (
            BLOCKING_DIR
            / "union"
            / "train_optimized_candidates.parquet"
        )
    )
)

S1_PATH = NORMALIZED_DIR / "train_s1.parquet"
S2_PATH = NORMALIZED_DIR / "train_s2.parquet"
S3_PATH = NORMALIZED_DIR / "train_s3.parquet"

FEATURE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

THREADS = 8
MEMORY_LIMIT = "8GB"
ROW_GROUP_SIZE = 250_000



def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def get_columns(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> set[str]:

    rows = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchall()

    return {row[0] for row in rows}


def count_rows(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> int:

    return con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchone()[0]


def output_path(source: str) -> Path:

    if source == "S2":
        return FEATURE_DIR / "train_features_s2.parquet"

    if source == "S3":
        return FEATURE_DIR / "train_features_s3.parquet"

    return FEATURE_DIR / "train_features.parquet"


def remove_file(path: Path) -> None:

    if path.exists():
        path.unlink()




def build_source_features(
    con: duckdb.DuckDBPyConnection,
    source: str,
    limit_per_source: int,
    output: Path,
    candidates_path: Path = CANDIDATES,
) -> int:

    if source not in {"S2", "S3"}:
        raise ValueError(
            f"Invalid source: {source}"
        )

    target_path = (
        S2_PATH
        if source == "S2"
        else S3_PATH
    )

    candidate_sql = sql_quote(candidates_path)
    s1_sql = sql_quote(S1_PATH)
    target_sql = sql_quote(target_path)
    output_sql = sql_quote(output)

    s1_columns = get_columns(con, S1_PATH)
    target_columns = get_columns(con, target_path)

    select_features = build_feature_select(
        s1_alias="s1",
        target_alias="t",
        candidate_alias="c",
        s1_columns=s1_columns,
        target_columns=target_columns,
    )

    limit_clause = ""

    if limit_per_source > 0:
        limit_clause = (
            f"\nLIMIT {int(limit_per_source)}"
        )

    print()
    print("=" * 80)
    print(f"BUILDING FEATURES FOR {source}")
    print("=" * 80)

    print()
    print(f"Candidate source : {source}")
    print(f"Target data      : {target_path}")
    print(f"Output           : {output}")

    if limit_per_source > 0:
        print(
            f"Mode             : SAMPLE "
            f"({limit_per_source:,} rows)"
        )
    else:
        print("Mode             : FULL DATASET")

    remove_file(output)

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
                FROM read_parquet('{candidate_sql}')

                WHERE matched_source = '{source}'

                {limit_clause}
            ) c

            INNER JOIN read_parquet('{s1_sql}') s1
                ON c.source1_entity_id = s1.entity_id

            INNER JOIN read_parquet('{target_sql}') t
                ON c.matched_entity_id = t.entity_id
        )

        TO '{output_sql}'

        (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            ROW_GROUP_SIZE {ROW_GROUP_SIZE}
        )
    """

    con.execute(query)

    rows = count_rows(
        con,
        output,
    )

    print()
    print(f"Rows written : {rows:,}")
    print(f"Output       : {output}")

    return rows


# ============================================================
# VALIDATE FEATURE FILE
# ============================================================

def validate_features(
    con: duckdb.DuckDBPyConnection,
    path: Path,
) -> None:

    print()
    print("=" * 80)
    print("VALIDATING FEATURES")
    print("=" * 80)

    columns = get_columns(
        con,
        path,
    )

    expected = set(
        feature_column_names()
    )

    missing = expected - columns

    if missing:
        raise RuntimeError(
            "Feature validation failed.\n"
            f"Missing columns: {sorted(missing)}"
        )

    extra = columns - expected

    if extra:
        print(
            "WARNING: extra columns found: "
            f"{sorted(extra)}"
        )

    rows = count_rows(
        con,
        path,
    )

    print()
    print(f"Rows : {rows:,}")

    # --------------------------------------------------------
    # Duplicate candidate check
    # --------------------------------------------------------

    duplicate_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM
        (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                COUNT(*) AS n
            FROM read_parquet('{sql_quote(path)}')
            GROUP BY
                source1_entity_id,
                matched_entity_id,
                matched_source
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    if duplicate_count != 0:
        raise RuntimeError(
            f"Duplicate candidate pairs detected: "
            f"{duplicate_count:,}"
        )

    print("Duplicate pairs : 0")

    # --------------------------------------------------------
    # NULL check for key features
    # --------------------------------------------------------

    nulls = con.execute(
        f"""
        SELECT
            COUNT(*) FILTER (
                WHERE name_exact IS NULL
            ) AS name_exact_nulls,

            COUNT(*) FILTER (
                WHERE address_exact IS NULL
            ) AS address_exact_nulls,

            COUNT(*) FILTER (
                WHERE country_exact IS NULL
            ) AS country_exact_nulls,

            COUNT(*) FILTER (
                WHERE name_char_ratio IS NULL
            ) AS name_ratio_nulls,

            COUNT(*) FILTER (
                WHERE address_char_ratio IS NULL
            ) AS address_ratio_nulls

        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchone()

    print()
    print("NULL checks:")
    print(f"  name_exact       : {nulls[0]:,}")
    print(f"  address_exact    : {nulls[1]:,}")
    print(f"  country_exact    : {nulls[2]:,}")
    print(f"  name_char_ratio  : {nulls[3]:,}")
    print(f"  address_char_ratio: {nulls[4]:,}")

    if any(value != 0 for value in nulls):
        raise RuntimeError(
            "NULLs detected in required numeric features."
        )

    # --------------------------------------------------------
    # Similarity range check
    # --------------------------------------------------------

    bad_ranges = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_quote(path)}')
        WHERE
            name_char_ratio < 0
            OR name_char_ratio > 1
            OR address_char_ratio < 0
            OR address_char_ratio > 1
            OR name_token_overlap < 0
            OR name_token_overlap > 1
            OR name_token_jaccard < 0
            OR name_token_jaccard > 1
            OR address_token_overlap < 0
            OR address_token_overlap > 1
            OR address_token_jaccard < 0
            OR address_token_jaccard > 1
            OR name_length_ratio < 0
            OR name_length_ratio > 1
            OR address_length_ratio < 0
            OR address_length_ratio > 1
        """
    ).fetchone()[0]

    print()
    print(
        f"Invalid similarity values : "
        f"{bad_ranges:,}"
    )

    if bad_ranges != 0:
        raise RuntimeError(
            "Similarity feature range validation failed."
        )

    # --------------------------------------------------------
    # Feature statistics
    # --------------------------------------------------------

    stats = con.execute(
        f"""
        SELECT
            AVG(name_exact),
            AVG(address_exact),
            AVG(country_exact),
            AVG(name_char_ratio),
            AVG(address_char_ratio),
            AVG(name_token_jaccard),
            AVG(address_token_jaccard),
            AVG(block_hybrid)
        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchone()

    print()
    print("Feature sanity statistics:")
    print(f"  name_exact           : {stats[0]:.6f}")
    print(f"  address_exact        : {stats[1]:.6f}")
    print(f"  country_exact        : {stats[2]:.6f}")
    print(f"  name_char_ratio      : {stats[3]:.6f}")
    print(f"  address_char_ratio   : {stats[4]:.6f}")
    print(f"  name_token_jaccard   : {stats[5]:.6f}")
    print(f"  address_token_jaccard: {stats[6]:.6f}")
    print(f"  block_hybrid         : {stats[7]:.6f}")

    print()
    print("FEATURE VALIDATION PASSED")


# ============================================================
# ARGUMENTS
# ============================================================

def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description=(
            "Build DuckDB-native pair features "
            "for Amazon ML Challenge 2026."
        )
    )

    parser.add_argument(
        "--source",
        choices=["S2", "S3", "all"],
        default="all",
        help="Source to process.",
    )

    parser.add_argument(
        "--sample-per-source",
        type=int,
        default=0,
        help=(
            "Rows per source. "
            "0 means full dataset."
        ),
    )

    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing final feature file.",
    )

    parser.add_argument(
        "--candidates",
        type=str,
        default="",
        help="Path to candidates parquet file.",
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default="",
        help="Prefix for output feature files (e.g. train_phase3_features).",
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    args = parse_args()

    cand_path = Path(args.candidates) if args.candidates else CANDIDATES
    out_prefix = args.output_prefix or ("train_phase3_features" if "phase3" in str(cand_path) else "train_features")

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("DUCKDB-NATIVE PAIR FEATURE ENGINE")
    print("=" * 80)

    # --------------------------------------------------------
    # Required files
    # --------------------------------------------------------

    required_files = [
        cand_path,
        S1_PATH,
        S2_PATH,
        S3_PATH,
    ]

    for path in required_files:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print()
    print(f"Candidates : {cand_path}")
    print(f"S1         : {S1_PATH}")
    print(f"S2         : {S2_PATH}")
    print(f"S3         : {S3_PATH}")
    print(f"Output dir : {FEATURE_DIR} (Prefix: {out_prefix})")

    con = duckdb.connect()

    try:

        # ----------------------------------------------------
        # Large-data configuration
        # ----------------------------------------------------

        con.execute(
            f"SET threads = {THREADS}"
        )

        con.execute(
            f"SET memory_limit = '{MEMORY_LIMIT}'"
        )

        con.execute(
            "SET preserve_insertion_order = false"
        )

        con.execute(
            "SET enable_progress_bar = true"
        )

        temp_dir = BLOCKING_DIR / "duckdb_tmp"
        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        con.execute(
            f"SET temp_directory = "
            f"'{sql_quote(temp_dir)}'"
        )

        # ----------------------------------------------------
        # Validate-only
        # ----------------------------------------------------

        if args.validate_only:

            path = FEATURE_DIR / f"{out_prefix}.parquet"

            if not path.exists():
                raise FileNotFoundError(
                    f"Feature file not found:\n{path}"
                )

            validate_features(
                con,
                path,
            )

            return

        # ----------------------------------------------------
        # SAMPLE MODE
        # ----------------------------------------------------

        if args.sample_per_source > 0:

            print()
            print(
                "RUNNING SAMPLE FEATURE GENERATION"
            )

            s2_output = (
                FEATURE_DIR
                / f"{out_prefix}_sample_s2.parquet"
            )

            s3_output = (
                FEATURE_DIR
                / f"{out_prefix}_sample_s3.parquet"
            )

            build_source_features(
                con,
                "S2",
                args.sample_per_source,
                s2_output,
                candidates_path=cand_path,
            )

            build_source_features(
                con,
                "S3",
                args.sample_per_source,
                s3_output,
                candidates_path=cand_path,
            )

            print()
            print("=" * 80)
            print("SAMPLE FEATURE GENERATION COMPLETE")
            print("=" * 80)

            print()
            print(
                "Files created:"
            )
            print(s2_output)
            print(s3_output)

            return

        # ----------------------------------------------------
        # FULL MODE
        # ----------------------------------------------------

        print()
        print(
            "RUNNING FULL FEATURE GENERATION"
        )

        s2_output = FEATURE_DIR / f"{out_prefix}_s2.parquet"
        s3_output = FEATURE_DIR / f"{out_prefix}_s3.parquet"

        if args.source in {"S2", "all"}:
            build_source_features(
                con,
                "S2",
                0,
                s2_output,
                candidates_path=cand_path,
            )

        if args.source in {"S3", "all"}:
            build_source_features(
                con,
                "S3",
                0,
                s3_output,
                candidates_path=cand_path,
            )

        # ----------------------------------------------------
        # Combine S2 + S3
        # ----------------------------------------------------

        if args.source == "all":

            final_output = (
                FEATURE_DIR
                / f"{out_prefix}.parquet"
            )

            remove_file(final_output)

            print()
            print("=" * 80)
            print("COMBINING S2 + S3 FEATURES")
            print("=" * 80)

            con.execute(
                f"""
                COPY
                (
                    SELECT *
                    FROM read_parquet(
                        [
                            '{sql_quote(s2_output)}',
                            '{sql_quote(s3_output)}'
                        ]
                    )
                )
                TO '{sql_quote(final_output)}'
                (
                    FORMAT PARQUET,
                    COMPRESSION SNAPPY,
                    ROW_GROUP_SIZE {ROW_GROUP_SIZE}
                )
                """
            )

            print()
            print(
                f"Final feature rows : "
                f"{count_rows(con, final_output):,}"
            )

            # ------------------------------------------------
            # Validate final file
            # ------------------------------------------------

            validate_features(
                con,
                final_output,
            )

            print()
            print("=" * 80)
            print("FULL FEATURE ENGINE COMPLETE")
            print("=" * 80)

            print()
            print("FINAL FILE:")
            print(final_output)

    finally:
        con.close()


if __name__ == "__main__":
    main()

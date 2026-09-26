from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "artifacts" / "features"

S2_FILE = FEATURE_DIR / "train_features_sample_s2.parquet"
S3_FILE = FEATURE_DIR / "train_features_sample_s3.parquet"

THREADS = 8
MEMORY_LIMIT = "4GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def validate_file(
    con: duckdb.DuckDBPyConnection,
    path: Path,
    expected_source: str,
) -> None:

    print()
    print("=" * 80)
    print(f"VALIDATING {expected_source} SAMPLE")
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(
            f"Feature sample not found:\n{path}"
        )

    file_sql = sql_quote(path)

    # --------------------------------------------------------
    # Schema
    # --------------------------------------------------------

    columns = [
        row[0]
        for row in con.execute(
            f"""
            DESCRIBE
            SELECT *
            FROM read_parquet('{file_sql}')
            """
        ).fetchall()
    ]

    required = {
        "source1_entity_id",
        "matched_entity_id",
        "matched_source",

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
    }

    missing = required - set(columns)

    if missing:
        raise RuntimeError(
            "Missing feature columns:\n"
            + "\n".join(sorted(missing))
        )

    print()
    print("Schema check : PASS")
    print(f"Columns      : {len(columns)}")

    # --------------------------------------------------------
    # Row count
    # --------------------------------------------------------

    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{file_sql}')
        """
    ).fetchone()[0]

    print(f"Rows         : {row_count:,}")

    if row_count != 5000:
        raise RuntimeError(
            f"Expected 5,000 rows, got {row_count:,}"
        )

    # --------------------------------------------------------
    # Source check
    # --------------------------------------------------------

    wrong_source = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{file_sql}')
        WHERE matched_source <> '{expected_source}'
        """
    ).fetchone()[0]

    print(
        f"Source check : "
        f"{'PASS' if wrong_source == 0 else 'FAIL'}"
    )

    if wrong_source != 0:
        raise RuntimeError(
            f"Found {wrong_source:,} wrong-source rows."
        )

    # --------------------------------------------------------
    # Duplicate pair check
    # --------------------------------------------------------

    duplicate_groups = con.execute(
        f"""
        SELECT COUNT(*)
        FROM
        (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source
            FROM read_parquet('{file_sql}')
            GROUP BY
                source1_entity_id,
                matched_entity_id,
                matched_source
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    print(
        f"Duplicate check: "
        f"{'PASS' if duplicate_groups == 0 else 'FAIL'}"
    )

    if duplicate_groups != 0:
        raise RuntimeError(
            f"Duplicate candidate groups: "
            f"{duplicate_groups:,}"
        )

    # --------------------------------------------------------
    # NULL checks
    # --------------------------------------------------------

    null_counts = con.execute(
        f"""
        SELECT
            COUNT(*) FILTER (
                WHERE name_exact IS NULL
            ),

            COUNT(*) FILTER (
                WHERE address_exact IS NULL
            ),

            COUNT(*) FILTER (
                WHERE country_exact IS NULL
            ),

            COUNT(*) FILTER (
                WHERE name_char_ratio IS NULL
            ),

            COUNT(*) FILTER (
                WHERE address_char_ratio IS NULL
            ),

            COUNT(*) FILTER (
                WHERE name_token_jaccard IS NULL
            ),

            COUNT(*) FILTER (
                WHERE address_token_jaccard IS NULL
            )

        FROM read_parquet('{file_sql}')
        """
    ).fetchone()

    print()
    print("NULL checks:")

    labels = [
        "name_exact",
        "address_exact",
        "country_exact",
        "name_char_ratio",
        "address_char_ratio",
        "name_token_jaccard",
        "address_token_jaccard",
    ]

    for label, value in zip(labels, null_counts):
        print(f"  {label:<24} {value:,}")

    if any(value != 0 for value in null_counts):
        raise RuntimeError(
            "NULL values detected in required features."
        )

    # --------------------------------------------------------
    # Similarity range checks
    # --------------------------------------------------------

    invalid_similarity = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{file_sql}')
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
        f"Similarity range check: "
        f"{'PASS' if invalid_similarity == 0 else 'FAIL'}"
    )

    if invalid_similarity != 0:
        raise RuntimeError(
            f"Invalid similarity rows: "
            f"{invalid_similarity:,}"
        )

    # --------------------------------------------------------
    # Binary feature checks
    # --------------------------------------------------------

    invalid_binary = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{file_sql}')
        WHERE
            name_exact NOT IN (0, 1)
            OR name_compact_exact NOT IN (0, 1)
            OR name_ascii_exact NOT IN (0, 1)

            OR address_exact NOT IN (0, 1)
            OR address_compact_exact NOT IN (0, 1)
            OR address_ascii_exact NOT IN (0, 1)

            OR country_exact NOT IN (0, 1)

            OR block_address NOT IN (0, 1)
            OR block_address_compact NOT IN (0, 1)
            OR block_name NOT IN (0, 1)
            OR block_rare_name NOT IN (0, 1)
            OR block_rare_address NOT IN (0, 1)
            OR block_hybrid NOT IN (0, 1)
        """
    ).fetchone()[0]

    print(
        f"Binary feature check: "
        f"{'PASS' if invalid_binary == 0 else 'FAIL'}"
    )

    if invalid_binary != 0:
        raise RuntimeError(
            f"Invalid binary feature rows: "
            f"{invalid_binary:,}"
        )

    # --------------------------------------------------------
    # Basic feature statistics
    # --------------------------------------------------------

    stats = con.execute(
        f"""
        SELECT
            AVG(name_exact),
            AVG(name_compact_exact),
            AVG(name_char_ratio),
            AVG(name_token_jaccard),

            AVG(address_exact),
            AVG(address_compact_exact),
            AVG(address_char_ratio),
            AVG(address_token_jaccard),

            AVG(country_exact),
            AVG(num_blocking_methods),
            AVG(block_hybrid)

        FROM read_parquet('{file_sql}')
        """
    ).fetchone()

    print()
    print("Feature sanity statistics:")
    print(f"  name_exact             : {stats[0]:.6f}")
    print(f"  name_compact_exact     : {stats[1]:.6f}")
    print(f"  name_char_ratio        : {stats[2]:.6f}")
    print(f"  name_token_jaccard     : {stats[3]:.6f}")
    print(f"  address_exact          : {stats[4]:.6f}")
    print(f"  address_compact_exact  : {stats[5]:.6f}")
    print(f"  address_char_ratio     : {stats[6]:.6f}")
    print(f"  address_token_jaccard  : {stats[7]:.6f}")
    print(f"  country_exact          : {stats[8]:.6f}")
    print(f"  avg_blocking_methods   : {stats[9]:.6f}")
    print(f"  block_hybrid            : {stats[10]:.6f}")

    print()
    print("=" * 80)
    print(f"{expected_source} SAMPLE VALIDATION PASSED")
    print("=" * 80)


def main() -> None:

    print("=" * 80)
    print("FEATURE SAMPLE VALIDATION")
    print("=" * 80)

    con = duckdb.connect()

    try:

        con.execute(f"SET threads = {THREADS}")
        con.execute(
            f"SET memory_limit = '{MEMORY_LIMIT}'"
        )

        con.execute(
            "SET preserve_insertion_order = false"
        )

        temp_dir = ROOT / "artifacts" / "blocking" / "duckdb_tmp"
        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        con.execute(
            f"SET temp_directory = "
            f"'{sql_quote(temp_dir)}'"
        )

        validate_file(
            con,
            S2_FILE,
            "S2",
        )

        validate_file(
            con,
            S3_FILE,
            "S3",
        )

        print()
        print("=" * 80)
        print("ALL SAMPLE VALIDATIONS PASSED")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
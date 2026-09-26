from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "artifacts"
    / "features"
    / "train_features.parquet"
)

EXPECTED_ROWS = 31_223_572

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def main() -> None:

    print("=" * 80)
    print("FULL TRAIN FEATURE VALIDATION")
    print("=" * 80)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    con = duckdb.connect()

    try:

        con.execute(f"SET threads = {THREADS}")
        con.execute(
            f"SET memory_limit = '{MEMORY_LIMIT}'"
        )
        con.execute(
            "SET preserve_insertion_order = false"
        )

        temp_dir = (
            ROOT
            / "artifacts"
            / "blocking"
            / "duckdb_tmp"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        con.execute(
            f"SET temp_directory = "
            f"'{sql_quote(temp_dir)}'"
        )

        feature_sql = sql_quote(FEATURE_FILE)

        # ----------------------------------------------------
        # ROW COUNT
        # ----------------------------------------------------

        row_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{feature_sql}')
            """
        ).fetchone()[0]

        print()
        print(f"Rows found    : {row_count:,}")
        print(f"Rows expected : {EXPECTED_ROWS:,}")

        if row_count != EXPECTED_ROWS:
            raise RuntimeError(
                f"Row-count mismatch: "
                f"{row_count:,} != {EXPECTED_ROWS:,}"
            )

        print("Row count     : PASS")

        # ----------------------------------------------------
        # SOURCE DISTRIBUTION
        # ----------------------------------------------------

        print()
        print("SOURCE DISTRIBUTION")

        source_rows = con.execute(
            f"""
            SELECT
                matched_source,
                COUNT(*) AS n
            FROM read_parquet('{feature_sql}')
            GROUP BY matched_source
            ORDER BY matched_source
            """
        ).fetchall()

        for source, count in source_rows:
            print(f"  {source}: {count:,}")

        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        print()
        print("CHECKING DUPLICATE PAIRS")

        duplicate_groups = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source
                FROM read_parquet('{feature_sql}')
                GROUP BY
                    source1_entity_id,
                    matched_entity_id,
                    matched_source
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        print(
            f"Duplicate groups : "
            f"{duplicate_groups:,}"
        )

        if duplicate_groups != 0:
            raise RuntimeError(
                "Duplicate candidate pairs detected."
            )

        # ----------------------------------------------------
        # NULL CHECK
        # ----------------------------------------------------

        print()
        print("CHECKING NULLS")

        nulls = con.execute(
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
            FROM read_parquet('{feature_sql}')
            """
        ).fetchone()

        names = [
            "name_exact",
            "address_exact",
            "country_exact",
            "name_char_ratio",
            "address_char_ratio",
            "name_token_jaccard",
            "address_token_jaccard",
        ]

        for name, value in zip(names, nulls):
            print(f"  {name:<25}: {value:,}")

        if any(value != 0 for value in nulls):
            raise RuntimeError(
                "NULL feature values detected."
            )

        # ----------------------------------------------------
        # RANGE CHECK
        # ----------------------------------------------------

        print()
        print("CHECKING SIMILARITY RANGES")

        invalid = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{feature_sql}')
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

        print(f"Invalid rows : {invalid:,}")

        if invalid != 0:
            raise RuntimeError(
                "Invalid similarity values detected."
            )

        # ----------------------------------------------------
        # BLOCKING DISTRIBUTION
        # ----------------------------------------------------

        print()
        print("BLOCKING DISTRIBUTION")

        blocking = con.execute(
            f"""
            SELECT
                num_blocking_methods,
                COUNT(*)
            FROM read_parquet('{feature_sql}')
            GROUP BY num_blocking_methods
            ORDER BY num_blocking_methods
            """
        ).fetchall()

        for methods, count in blocking:
            print(
                f"  {methods} method(s): "
                f"{count:,}"
            )

        # ----------------------------------------------------
        # FEATURE STATISTICS
        # ----------------------------------------------------

        print()
        print("GLOBAL FEATURE STATISTICS")

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
                AVG(block_hybrid)

            FROM read_parquet('{feature_sql}')
            """
        ).fetchone()

        labels = [
            "name_exact",
            "name_compact_exact",
            "name_char_ratio",
            "name_token_jaccard",
            "address_exact",
            "address_compact_exact",
            "address_char_ratio",
            "address_token_jaccard",
            "country_exact",
            "block_hybrid",
        ]

        for label, value in zip(labels, stats):
            print(
                f"  {label:<25}: "
                f"{value:.6f}"
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("FULL FEATURE VALIDATION PASSED")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
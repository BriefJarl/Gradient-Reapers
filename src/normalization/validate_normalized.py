from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"


FILES = {
    "train_s1": NORMALIZED_DIR / "train_s1.parquet",
    "train_s2": NORMALIZED_DIR / "train_s2.parquet",
    "train_s3": NORMALIZED_DIR / "train_s3.parquet",
    "test_s1": NORMALIZED_DIR / "test_s1.parquet",
    "test_s2": NORMALIZED_DIR / "test_s2.parquet",
    "test_s3": NORMALIZED_DIR / "test_s3.parquet",
}


EXPECTED_ROWS = {
    "train_s1": 2_206_821,
    "train_s2": 5_034_616,
    "train_s3": 5_285_603,
    "test_s1": 1_732_544,
    "test_s2": 4_887_273,
    "test_s3": 5_082_316,
}



def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")



def validate_file(
    con: duckdb.DuckDBPyConnection,
    name: str,
    path: Path,
) -> None:

    print("\n" + "=" * 80)
    print(f"VALIDATING: {name.upper()}")
    print("=" * 80)

    if not path.exists():
        raise FileNotFoundError(
            f"Missing normalized artifact:\n{path}"
        )

    p = sql_path(path)


    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{p}')
        """
    ).fetchone()[0]

    expected = EXPECTED_ROWS[name]

    print(f"Rows              : {row_count:,}")
    print(f"Expected rows     : {expected:,}")

    if row_count != expected:
        raise ValueError(
            f"{name}: row count mismatch. "
            f"Expected {expected:,}, got {row_count:,}"
        )


    duplicate_ids = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT entity_id
            FROM read_parquet('{p}')
            GROUP BY entity_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    empty_ids = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{p}')
        WHERE entity_id IS NULL
           OR trim(entity_id) = ''
        """
    ).fetchone()[0]

    print(f"Duplicate IDs     : {duplicate_ids:,}")
    print(f"Empty IDs         : {empty_ids:,}")

    if duplicate_ids != 0:
        raise ValueError(
            f"{name}: duplicate entity IDs found."
        )

    if empty_ids != 0:
        raise ValueError(
            f"{name}: empty entity IDs found."
        )


    stats = con.execute(
        f"""
        SELECT

            COUNT(*) FILTER (
                WHERE name_norm IS NULL
                   OR trim(name_norm) = ''
            ) AS empty_name_norm,

            COUNT(*) FILTER (
                WHERE name_compact IS NULL
                   OR trim(name_compact) = ''
            ) AS empty_name_compact,

            COUNT(*) FILTER (
                WHERE address_norm IS NULL
                   OR trim(address_norm) = ''
            ) AS empty_address_norm,

            COUNT(*) FILTER (
                WHERE address_compact IS NULL
                   OR trim(address_compact) = ''
            ) AS empty_address_compact,

            COUNT(*) FILTER (
                WHERE country_norm IS NULL
                   OR trim(country_norm) = ''
            ) AS empty_country_norm,

            COUNT(*) FILTER (
                WHERE name_tokens IS NULL
                   OR len(name_tokens) = 0
            ) AS empty_name_tokens,

            COUNT(*) FILTER (
                WHERE address_tokens IS NULL
                   OR len(address_tokens) = 0
            ) AS empty_address_tokens

        FROM read_parquet('{p}')
        """
    ).fetchone()

    (
        empty_name_norm,
        empty_name_compact,
        empty_address_norm,
        empty_address_compact,
        empty_country_norm,
        empty_name_tokens,
        empty_address_tokens,
    ) = stats

    print("\nNormalized-field checks:")
    print(f"Empty name_norm           : {empty_name_norm:,}")
    print(f"Empty name_compact        : {empty_name_compact:,}")
    print(f"Empty address_norm        : {empty_address_norm:,}")
    print(f"Empty address_compact     : {empty_address_compact:,}")
    print(f"Empty country_norm        : {empty_country_norm:,}")
    print(f"Empty name_tokens         : {empty_name_tokens:,}")
    print(f"Empty address_tokens      : {empty_address_tokens:,}")


    raw_stats = con.execute(
        f"""
        SELECT

            COUNT(*) FILTER (
                WHERE business_name IS NULL
                   OR trim(business_name) = ''
            ) AS missing_name,

            COUNT(*) FILTER (
                WHERE business_address IS NULL
                   OR trim(business_address) = ''
            ) AS missing_address,

            COUNT(*) FILTER (
                WHERE country IS NULL
                   OR trim(country) = ''
            ) AS missing_country

        FROM read_parquet('{p}')
        """
    ).fetchone()

    missing_name, missing_address, missing_country = raw_stats

    print("\nRaw-field checks:")
    print(f"Missing business_name      : {missing_name:,}")
    print(f"Missing business_address   : {missing_address:,}")
    print(f"Missing country            : {missing_country:,}")


    print("\nSample normalized records:")

    sample = con.execute(
        f"""
        SELECT
            entity_id,
            business_name,
            name_norm,
            name_compact,
            business_address,
            address_norm,
            address_compact,
            country,
            country_norm
        FROM read_parquet('{p}')
        LIMIT 5
        """
    ).fetchall()

    for row in sample:
        print(row)

    print("\nValidation: PASSED")

def main() -> None:

    print("\n" + "=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("NORMALIZATION QA")
    print("=" * 80)

    con = duckdb.connect()

    try:

        for name, path in FILES.items():
            validate_file(
                con,
                name,
                path,
            )

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("ALL NORMALIZATION CHECKS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

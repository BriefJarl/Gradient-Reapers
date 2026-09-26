from __future__ import annotations

from pathlib import Path

import duckdb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"

INDEX_DIR = BLOCKING_DIR / "indexes"

INDEX_DIR.mkdir(parents=True, exist_ok=True)


SOURCES = {
    "S2": NORMALIZED_DIR / "train_s2.parquet",
    "S3": NORMALIZED_DIR / "train_s3.parquet",
}


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


# ============================================================
# BASIC INDEX
# ============================================================

def build_basic_index(
    con: duckdb.DuckDBPyConnection,
    source: str,
    path: Path,
) -> None:

    print("\n" + "=" * 80)
    print(f"BUILDING BASIC BLOCKING INDEX: {source}")
    print("=" * 80)

    p = sql_path(path)

    # --------------------------------------------------------
    # Keep only fields needed by blocking.
    # --------------------------------------------------------

    output = INDEX_DIR / f"{source.lower()}_blocking_base.parquet"
    output_sql = sql_path(output)

    con.execute(
        f"""
        COPY (
            SELECT
                entity_id,
                country_norm,
                name_norm,
                name_compact,
                name_tokens,
                name_numeric_tokens,
                address_norm,
                address_compact,
                address_tokens,
                address_numeric_tokens
            FROM read_parquet('{p}')
        )
        TO '{output_sql}'
        (FORMAT PARQUET);
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    print(f"Output : {output}")
    print(f"Rows   : {count:,}")


# ============================================================
# ADDRESS BLOCKING INDEX
# ============================================================

def build_address_index(
    con: duckdb.DuckDBPyConnection,
    source: str,
    path: Path,
) -> None:

    print("\n" + "-" * 80)
    print(f"ADDRESS INDEX: {source}")
    print("-" * 80)

    p = sql_path(path)

    output = INDEX_DIR / f"{source.lower()}_address_index.parquet"
    output_sql = sql_path(output)

    con.execute(
        f"""
        COPY (
            SELECT
                country_norm,
                address_norm AS block_key,
                entity_id
            FROM read_parquet('{p}')
            WHERE
                country_norm <> ''
                AND address_norm <> ''
        )
        TO '{output_sql}'
        (FORMAT PARQUET);
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    distinct_keys = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT DISTINCT
                country_norm,
                block_key
            FROM read_parquet('{output_sql}')
        )
        """
    ).fetchone()[0]

    print(f"Rows           : {count:,}")
    print(f"Distinct keys  : {distinct_keys:,}")
    print(f"Output         : {output}")


# ============================================================
# COMPACT ADDRESS INDEX
# ============================================================

def build_compact_address_index(
    con: duckdb.DuckDBPyConnection,
    source: str,
    path: Path,
) -> None:

    print("\n" + "-" * 80)
    print(f"COMPACT ADDRESS INDEX: {source}")
    print("-" * 80)

    p = sql_path(path)

    output = INDEX_DIR / f"{source.lower()}_address_compact_index.parquet"
    output_sql = sql_path(output)

    con.execute(
        f"""
        COPY (
            SELECT
                country_norm,
                address_compact AS block_key,
                entity_id
            FROM read_parquet('{p}')
            WHERE
                country_norm <> ''
                AND address_compact <> ''
        )
        TO '{output_sql}'
        (FORMAT PARQUET);
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    print(f"Rows           : {count:,}")
    print(f"Output         : {output}")


# ============================================================
# EXACT NAME INDEX
# ============================================================

def build_name_index(
    con: duckdb.DuckDBPyConnection,
    source: str,
    path: Path,
) -> None:

    print("\n" + "-" * 80)
    print(f"NAME INDEX: {source}")
    print("-" * 80)

    p = sql_path(path)

    output = INDEX_DIR / f"{source.lower()}_name_index.parquet"
    output_sql = sql_path(output)

    con.execute(
        f"""
        COPY (
            SELECT
                country_norm,
                name_norm AS block_key,
                entity_id
            FROM read_parquet('{p}')
            WHERE
                country_norm <> ''
                AND name_norm <> ''
        )
        TO '{output_sql}'
        (FORMAT PARQUET);
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    print(f"Rows           : {count:,}")
    print(f"Output         : {output}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("\n" + "=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("BLOCKING INDEX BUILDER")
    print("=" * 80)

    con = duckdb.connect()

    try:

        for source, path in SOURCES.items():

            if not path.exists():
                raise FileNotFoundError(
                    f"Missing normalized file: {path}"
                )

            build_basic_index(
                con,
                source,
                path,
            )

            build_address_index(
                con,
                source,
                path,
            )

            build_compact_address_index(
                con,
                source,
                path,
            )

            build_name_index(
                con,
                source,
                path,
            )

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("BLOCKING INDEX BUILD COMPLETE")
    print("=" * 80)

    print(f"\nIndexes created in:")
    print(INDEX_DIR)


if __name__ == "__main__":
    main()
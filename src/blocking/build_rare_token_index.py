from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
INDEX_DIR = BLOCKING_DIR / "indexes"

INDEX_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASETS
# ============================================================

TARGETS = {
    "S2": NORMALIZED_DIR / "train_s2.parquet",
    "S3": NORMALIZED_DIR / "train_s3.parquet",
}


# ============================================================
# RARE-TOKEN CONFIGURATION
# ============================================================

# A token appearing in at most this many target records
# is considered sufficiently selective for blocking.
#
# This is deliberately conservative because the datasets
# contain millions of records.
MAX_TOKEN_FREQ = 50

# We keep even one-character Unicode tokens if they are rare.
# This avoids accidentally damaging multilingual data.
MIN_TOKEN_LENGTH = 1


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    """
    Convert Windows paths into DuckDB-safe paths.
    """
    return str(path).replace("\\", "/")


def output_path(
    source_name: str,
    token_type: str,
) -> Path:
    """
    Build a deterministic output filename.
    """
    return (
        INDEX_DIR
        / f"train_{source_name.lower()}_rare_{token_type}_token_index.parquet"
    )


# ============================================================
# BUILD ONE RARE TOKEN INDEX
# ============================================================

def build_index(
    con: duckdb.DuckDBPyConnection,
    source_name: str,
    source_path: Path,
    token_type: str,
    token_column: str,
) -> Path:

    source_sql = sql_path(source_path)
    output = output_path(
        source_name,
        token_type,
    )
    output_sql = sql_path(output)

    print("\n" + "=" * 80)
    print(
        f"BUILDING RARE {token_type.upper()} TOKEN INDEX: "
        f"{source_name}"
    )
    print("=" * 80)

    print(f"Source : {source_path}")
    print(f"Output : {output}")
    print(f"Max token frequency : {MAX_TOKEN_FREQ}")

    # --------------------------------------------------------
    # Name tokens:
    #
    # Pure numeric tokens are excluded because numbers in
    # business names are generally poor standalone keys.
    #
    # Address tokens:
    #
    # Numeric tokens are retained because house/building
    # numbers can be highly informative.
    # --------------------------------------------------------

    numeric_filter = ""

    if token_type == "name":
        numeric_filter = """
            AND NOT regexp_matches(token, '^[0-9]+$')
        """

    query = f"""
        COPY (
            WITH exploded_tokens AS (

                SELECT DISTINCT
                    CAST(entity_id AS VARCHAR) AS entity_id,
                    country_norm,
                    LOWER(TRIM(token)) AS token

                FROM read_parquet('{source_sql}')

                CROSS JOIN UNNEST({token_column}) AS u(token)

                WHERE country_norm IS NOT NULL
                  AND TRIM(country_norm) <> ''

                  AND token IS NOT NULL
                  AND TRIM(token) <> ''

                  AND LENGTH(TRIM(token)) >= {MIN_TOKEN_LENGTH}

                  {numeric_filter}
            ),

            token_frequency AS (

                SELECT
                    country_norm,
                    token,
                    COUNT(*) AS token_freq

                FROM exploded_tokens

                GROUP BY
                    country_norm,
                    token

                HAVING COUNT(*) <= {MAX_TOKEN_FREQ}
            )

            SELECT
                e.entity_id,
                e.country_norm,
                e.token,
                f.token_freq

            FROM exploded_tokens AS e

            INNER JOIN token_frequency AS f
                ON e.country_norm = f.country_norm
               AND e.token = f.token

        )

        TO '{output_sql}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    con.execute(query)

    # --------------------------------------------------------
    # Validation statistics
    # --------------------------------------------------------

    total_rows = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    distinct_tokens = con.execute(
        f"""
        SELECT COUNT(DISTINCT token)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    distinct_entities = con.execute(
        f"""
        SELECT COUNT(DISTINCT entity_id)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    max_frequency = con.execute(
        f"""
        SELECT COALESCE(MAX(token_freq), 0)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    print()
    print(f"Index rows        : {total_rows:,}")
    print(f"Distinct tokens   : {distinct_tokens:,}")
    print(f"Distinct entities : {distinct_entities:,}")
    print(f"Max token freq    : {max_frequency:,}")
    print(f"Output            : {output}")

    # Safety assertion.
    if max_frequency > MAX_TOKEN_FREQ:
        raise RuntimeError(
            f"Unsafe rare-token index: max frequency "
            f"{max_frequency} exceeds configured limit "
            f"{MAX_TOKEN_FREQ}."
        )

    return output


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Build memory-safe rare-token blocking indexes."
    )

    parser.add_argument(
        "--token-type",
        choices=[
            "name",
            "address",
            "both",
        ],
        default="both",
        help="Which token index to build.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("RARE TOKEN INDEX BUILDER")
    print("=" * 80)

    con = duckdb.connect()

    try:

        for source_name, source_path in TARGETS.items():

            if args.token_type in {"name", "both"}:
                build_index(
                    con=con,
                    source_name=source_name,
                    source_path=source_path,
                    token_type="name",
                    token_column="name_tokens",
                )

            if args.token_type in {"address", "both"}:
                build_index(
                    con=con,
                    source_name=source_name,
                    source_path=source_path,
                    token_type="address",
                    token_column="address_tokens",
                )

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("RARE TOKEN INDEX BUILD COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
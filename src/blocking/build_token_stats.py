from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
OUTPUT_DIR = ROOT / "artifacts" / "blocking"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


SOURCES = {
    "S1": NORMALIZED_DIR / "train_s1.parquet",
    "S2": NORMALIZED_DIR / "train_s2.parquet",
    "S3": NORMALIZED_DIR / "train_s3.parquet",
}



def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def build_token_stats(
    con: duckdb.DuckDBPyConnection,
    source: str,
    path: Path,
) -> None:

    print("\n" + "=" * 80)
    print(f"BUILDING TOKEN STATISTICS: {source}")
    print("=" * 80)

    path_sql = sql_path(path)

    output_path = OUTPUT_DIR / f"{source.lower()}_token_stats.parquet"
    output_sql = sql_path(output_path)

    query = f"""
        COPY (
            WITH base AS (
                SELECT
                    entity_id,
                    country_norm,
                    name_tokens
                FROM read_parquet('{path_sql}')
            ),

            exploded AS (
                SELECT
                    entity_id,
                    country_norm,
                    trim(token) AS token
                FROM base
                CROSS JOIN UNNEST(name_tokens) AS t(token)
            ),

            deduplicated AS (
                SELECT DISTINCT
                    entity_id,
                    country_norm,
                    token
                FROM exploded
                WHERE
                    token <> ''
                    AND length(token) >= 2
            ),

            token_stats AS (
                SELECT
                    country_norm,
                    token,
                    COUNT(*) AS doc_frequency
                FROM deduplicated
                GROUP BY
                    country_norm,
                    token
            )

            SELECT
                '{source}' AS source,
                country_norm,
                token,
                doc_frequency,
                length(token) AS token_length
            FROM token_stats
            ORDER BY
                source,
                country_norm,
                doc_frequency,
                token
        )
        TO '{output_sql}'
        (FORMAT PARQUET);
    """

    con.execute(query)

    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    print(f"Output : {output_path}")
    print(f"Rows   : {row_count:,}")

    print("\nMost frequent tokens:")

    rows = con.execute(
        f"""
        SELECT
            country_norm,
            token,
            doc_frequency
        FROM read_parquet('{output_sql}')
        ORDER BY doc_frequency DESC
        LIMIT 25
        """
    ).fetchall()

    for country, token, frequency in rows:
        print(
            f"{country:10} | "
            f"{frequency:10,} | "
            f"{token}"
        )


def main() -> None:

    print("\nAMAZON ML CHALLENGE 2026")
    print("Blocking Token Statistics")
    print("=" * 80)

    con = duckdb.connect()

    try:
        for source, path in SOURCES.items():

            if not path.exists():
                raise FileNotFoundError(
                    f"Normalized dataset not found: {path}"
                )

            build_token_stats(
                con=con,
                source=source,
                path=path,
            )

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("TOKEN STATISTICS COMPLETE")
    print("=" * 80)

    print(f"\nArtifacts created in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

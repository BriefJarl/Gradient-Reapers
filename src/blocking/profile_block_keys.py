from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"


SOURCES = {
    "S1": NORMALIZED_DIR / "train_s1.parquet",
    "S2": NORMALIZED_DIR / "train_s2.parquet",
    "S3": NORMALIZED_DIR / "train_s3.parquet",
}


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def profile_key(
    con: duckdb.DuckDBPyConnection,
    source: str,
    path: Path,
    expression: str,
    block_name: str,
) -> None:

    p = sql_path(path)

    print("\n" + "=" * 80)
    print(f"{source} | {block_name}")
    print("=" * 80)

    query = f"""
        WITH buckets AS (
            SELECT
                country_norm,
                {expression} AS block_key,
                COUNT(*) AS bucket_size
            FROM read_parquet('{p}')
            WHERE
                country_norm <> ''
                AND {expression} <> ''
            GROUP BY
                country_norm,
                {expression}
        )

        SELECT
            COUNT(*) AS number_of_buckets,

            SUM(bucket_size) AS rows_covered,

            AVG(bucket_size) AS mean_bucket,

            quantile_cont(bucket_size, 0.50) AS p50,

            quantile_cont(bucket_size, 0.90) AS p90,

            quantile_cont(bucket_size, 0.95) AS p95,

            quantile_cont(bucket_size, 0.99) AS p99,

            MAX(bucket_size) AS max_bucket

        FROM buckets
    """

    result = con.execute(query).fetchone()

    (
        number_of_buckets,
        rows_covered,
        mean_bucket,
        p50,
        p90,
        p95,
        p99,
        max_bucket,
    ) = result

    print(f"Buckets       : {number_of_buckets:,}")
    print(f"Rows covered  : {rows_covered:,}")
    print(f"Mean          : {mean_bucket:.2f}")
    print(f"P50           : {p50:.2f}")
    print(f"P90           : {p90:.2f}")
    print(f"P95           : {p95:.2f}")
    print(f"P99           : {p99:.2f}")
    print(f"MAX           : {max_bucket:,}")


def main() -> None:

    print("\n" + "=" * 80)
    print("BLOCK KEY BUCKET PROFILING")
    print("=" * 80)

    con = duckdb.connect()

    try:

        for source, path in SOURCES.items():

            profile_key(
                con,
                source,
                path,
                "address_norm",
                "COUNTRY + ADDRESS_NORM",
            )

            profile_key(
                con,
                source,
                path,
                "address_compact",
                "COUNTRY + ADDRESS_COMPACT",
            )

            profile_key(
                con,
                source,
                path,
                "name_norm",
                "COUNTRY + NAME_NORM",
            )

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("BLOCK KEY PROFILING COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
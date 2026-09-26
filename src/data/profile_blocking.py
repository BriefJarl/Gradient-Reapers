from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"
ARTIFACT_DIR = ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)


S1 = TRAIN_DIR / "train_source1.tsv"
S2 = TRAIN_DIR / "train_source2.tsv"
S3 = TRAIN_DIR / "train_source3.tsv"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def normalized_name(expr: str) -> str:
    """
    Conservative normalization for profiling only.

    We deliberately do NOT remove linguistic information aggressively.
    The goal here is to measure blocking potential.
    """
    return f"""
        lower(
            trim(
                regexp_replace(
                    {expr},
                    '[^[:alnum:][:space:]]',
                    '',
                    'g'
                )
            )
        )
    """


def normalized_address(expr: str) -> str:
    return f"""
        lower(
            trim(
                regexp_replace(
                    {expr},
                    '[^[:alnum:][:space:]]',
                    '',
                    'g'
                )
            )
        )
    """


def profile_source(con, path: Path, source_name: str):
    p = sql_path(path)

    print("\n" + "=" * 75)
    print(f"{source_name} BLOCKING PROFILE")
    print("=" * 75)

    table = f"""
        read_csv_auto(
            '{p}',
            delim='\\t',
            header=true,
            sample_size=10000
        )
    """

    name_norm = normalized_name("business_name")
    address_norm = normalized_address("business_address")

    # ------------------------------------------------------------
    # NAME CARDINALITY
    # ------------------------------------------------------------

    print("\n[1] BUSINESS NAME CARDINALITY")

    result = con.execute(
        f"""
        WITH base AS (
            SELECT
                business_name,
                country,
                {name_norm} AS name_norm
            FROM {table}
        )
        SELECT
            COUNT(*) AS total_rows,
            COUNT(DISTINCT name_norm) AS unique_names,
            COUNT(*) - COUNT(DISTINCT name_norm) AS duplicate_rows
        FROM base
        """
    ).fetchone()

    total, unique_names, duplicate_rows = result

    print(f"Total rows          : {total:,}")
    print(f"Unique normalized   : {unique_names:,}")
    print(f"Duplicate rows      : {duplicate_rows:,}")
    print(
        f"Unique ratio        : "
        f"{unique_names / total * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # NAME FREQUENCY DISTRIBUTION
    # ------------------------------------------------------------

    print("\n[2] NAME FREQUENCY DISTRIBUTION")

    result = con.execute(
        f"""
        WITH base AS (
            SELECT
                {name_norm} AS name_norm
            FROM {table}
        ),
        freq AS (
            SELECT
                name_norm,
                COUNT(*) AS cnt
            FROM base
            WHERE name_norm <> ''
            GROUP BY name_norm
        )
        SELECT
            COUNT(*) FILTER (WHERE cnt = 1) AS freq_1,
            COUNT(*) FILTER (WHERE cnt = 2) AS freq_2,
            COUNT(*) FILTER (WHERE cnt BETWEEN 3 AND 5) AS freq_3_5,
            COUNT(*) FILTER (WHERE cnt BETWEEN 6 AND 10) AS freq_6_10,
            COUNT(*) FILTER (WHERE cnt BETWEEN 11 AND 50) AS freq_11_50,
            COUNT(*) FILTER (WHERE cnt > 50) AS freq_gt_50,
            MAX(cnt) AS max_frequency
        FROM freq
        """
    ).fetchone()

    labels = [
        "exactly 1",
        "2",
        "3-5",
        "6-10",
        "11-50",
        ">50",
    ]

    for label, value in zip(labels, result[:-1]):
        print(f"Names occurring {label:>8}: {value:,}")

    print(f"Maximum name frequency : {result[-1]:,}")

    # ------------------------------------------------------------
    # ADDRESS CARDINALITY
    # ------------------------------------------------------------

    print("\n[3] BUSINESS ADDRESS CARDINALITY")

    result = con.execute(
        f"""
        WITH base AS (
            SELECT
                {address_norm} AS address_norm
            FROM {table}
            WHERE business_address IS NOT NULL
        )
        SELECT
            COUNT(*) AS non_null_rows,
            COUNT(DISTINCT address_norm) AS unique_addresses,
            COUNT(*) - COUNT(DISTINCT address_norm) AS duplicate_rows
        FROM base
        """
    ).fetchone()

    non_null, unique_addresses, duplicate_addresses = result

    print(f"Non-null rows        : {non_null:,}")
    print(f"Unique normalized    : {unique_addresses:,}")
    print(f"Duplicate rows       : {duplicate_addresses:,}")

    print(
        f"Unique ratio         : "
        f"{unique_addresses / non_null * 100:.2f}%"
    )

    # ------------------------------------------------------------
    # COUNTRY + NAME
    # ------------------------------------------------------------

    print("\n[4] COUNTRY + NAME BLOCK QUALITY")

    result = con.execute(
        f"""
        WITH base AS (
            SELECT
                country,
                {name_norm} AS name_norm
            FROM {table}
        ),
        freq AS (
            SELECT
                country,
                name_norm,
                COUNT(*) AS cnt
            FROM base
            WHERE name_norm <> ''
            GROUP BY country, name_norm
        )
        SELECT
            COUNT(*) AS distinct_keys,
            SUM(cnt) AS rows,
            SUM(
                CASE
                    WHEN cnt = 1 THEN 1
                    ELSE 0
                END
            ) AS singleton_keys,
            MAX(cnt) AS max_bucket
        FROM freq
        """
    ).fetchone()

    distinct_keys, rows, singleton_keys, max_bucket = result

    print(f"Distinct country+name keys : {distinct_keys:,}")
    print(f"Rows covered               : {rows:,}")
    print(f"Singleton keys             : {singleton_keys:,}")
    print(f"Largest bucket             : {max_bucket:,}")

    # ------------------------------------------------------------
    # COUNTRY + ADDRESS
    # ------------------------------------------------------------

    print("\n[5] COUNTRY + ADDRESS BLOCK QUALITY")

    result = con.execute(
        f"""
        WITH base AS (
            SELECT
                country,
                {address_norm} AS address_norm
            FROM {table}
            WHERE business_address IS NOT NULL
        ),
        freq AS (
            SELECT
                country,
                address_norm,
                COUNT(*) AS cnt
            FROM base
            WHERE address_norm <> ''
            GROUP BY country, address_norm
        )
        SELECT
            COUNT(*) AS distinct_keys,
            SUM(cnt) AS rows,
            SUM(
                CASE
                    WHEN cnt = 1 THEN 1
                    ELSE 0
                END
            ) AS singleton_keys,
            MAX(cnt) AS max_bucket
        FROM freq
        """
    ).fetchone()

    distinct_keys, rows, singleton_keys, max_bucket = result

    print(f"Distinct country+address keys : {distinct_keys:,}")
    print(f"Rows covered                   : {rows:,}")
    print(f"Singleton keys                 : {singleton_keys:,}")
    print(f"Largest bucket                 : {max_bucket:,}")

    # ------------------------------------------------------------
    # TOP NAME COLLISIONS
    # ------------------------------------------------------------

    print("\n[6] TOP NORMALIZED NAME COLLISIONS")

    rows = con.execute(
        f"""
        SELECT
            {name_norm} AS name_norm,
            country,
            COUNT(*) AS cnt
        FROM {table}
        WHERE business_name IS NOT NULL
        GROUP BY name_norm, country
        HAVING COUNT(*) > 1
        ORDER BY cnt DESC
        LIMIT 15
        """
    ).fetchall()

    for name, country, count in rows:
        print(
            f"{count:8,} | {str(country):8} | {name[:60]}"
        )


def main():
    con = duckdb.connect()

    print("\nAMAZON ML CHALLENGE 2026")
    print("Blocking Intelligence Profiler")

    profile_source(con, S1, "SOURCE 1")
    profile_source(con, S2, "SOURCE 2")
    profile_source(con, S3, "SOURCE 3")

    print("\n" + "=" * 75)
    print("BLOCKING PROFILING COMPLETE")
    print("=" * 75)


if __name__ == "__main__":
    main()
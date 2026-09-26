from pathlib import Path
import json
import duckdb


ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"
TEST_DIR = ROOT / "student_resource" / "dataset" / "test"
ARTIFACT_DIR = ROOT / "artifacts"

ARTIFACT_DIR.mkdir(exist_ok=True)


def query_file(con, path, query):
    path = str(path).replace("\\", "/")

    return con.execute(
        query.replace("{FILE}", path)
    ).fetchall()


def profile_dataset(con, path):
    print(f"\n{'=' * 70}")
    print(f"FILE: {path.name}")
    print(f"{'=' * 70}")

    path_sql = str(path).replace("\\", "/")

    # Row count
    row_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_csv_auto(
            '{path_sql}',
            delim='\\t',
            header=true,
            sample_size=10000
        )
        """
    ).fetchone()[0]

    print(f"Rows: {row_count:,}")

    # Columns
    columns = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_csv_auto(
            '{path_sql}',
            delim='\\t',
            header=true,
            sample_size=10000
        )
        """
    ).fetchall()

    print("\nColumns:")

    for col in columns:
        print(f"  {col[0]:25} {col[1]}")

    # Missingness
    missing = con.execute(
        f"""
        SELECT
            COUNT(*) AS total_rows,

            SUM(
                CASE
                    WHEN business_name IS NULL
                         OR TRIM(business_name) = ''
                    THEN 1
                    ELSE 0
                END
            ) AS missing_name,

            SUM(
                CASE
                    WHEN business_address IS NULL
                         OR TRIM(business_address) = ''
                    THEN 1
                    ELSE 0
                END
            ) AS missing_address,

            SUM(
                CASE
                    WHEN country IS NULL
                         OR TRIM(country) = ''
                    THEN 1
                    ELSE 0
                END
            ) AS missing_country

        FROM read_csv_auto(
            '{path_sql}',
            delim='\\t',
            header=true,
            sample_size=10000
        )
        """
    ).fetchone()

    total, missing_name, missing_address, missing_country = missing

    print("\nMissing values:")

    print(
        f"  business_name     {missing_name:,} "
        f"({missing_name / total * 100:.3f}%)"
    )

    print(
        f"  business_address  {missing_address:,} "
        f"({missing_address / total * 100:.3f}%)"
    )

    print(
        f"  country           {missing_country:,} "
        f"({missing_country / total * 100:.3f}%)"
    )

    # Country distribution
    print("\nTop countries:")

    countries = con.execute(
        f"""
        SELECT
            country,
            COUNT(*) AS count
        FROM read_csv_auto(
            '{path_sql}',
            delim='\\t',
            header=true,
            sample_size=10000
        )
        GROUP BY country
        ORDER BY count DESC
        LIMIT 20
        """
    ).fetchall()

    for country, count in countries:
        print(f"  {str(country):20} {count:,}")


def profile_ground_truth(con, path):
    print(f"\n{'=' * 70}")
    print("GROUND TRUTH ANALYSIS")
    print(f"{'=' * 70}")

    path_sql = str(path).replace("\\", "/")

    query = f"""
        WITH gt AS (
            SELECT
                source1_entity_id,
                matched_entity_ids
            FROM read_csv_auto(
                '{path_sql}',
                delim='\\t',
                header=true,
                sample_size=10000
            )
        ),
        counts AS (
            SELECT
                source1_entity_id,
                CASE
                    WHEN matched_entity_ids IS NULL
                         OR TRIM(matched_entity_ids) = ''
                    THEN 0
                    ELSE
                        LENGTH(matched_entity_ids)
                        - LENGTH(REPLACE(matched_entity_ids, ',', ''))
                        + 1
                END AS match_count
            FROM gt
        )
        SELECT
            match_count,
            COUNT(*) AS entity_count
        FROM counts
        GROUP BY match_count
        ORDER BY match_count
    """

    rows = con.execute(query).fetchall()

    total_entities = sum(row[1] for row in rows)

    print("\nMatches per Source-1 entity:")

    for match_count, entity_count in rows:
        percentage = (
            entity_count / total_entities * 100
            if total_entities
            else 0
        )

        print(
            f"  {match_count:3} matches -> "
            f"{entity_count:,} S1 entities "
            f"({percentage:.3f}%)"
        )

    print(f"\nTotal Source-1 entities: {total_entities:,}")


def main():
    con = duckdb.connect()

    print("\nAMAZON ML CHALLENGE 2026")
    print("Large-scale Entity Resolution Profiler")

    train_files = [
        TRAIN_DIR / "train_source1.tsv",
        TRAIN_DIR / "train_source2.tsv",
        TRAIN_DIR / "train_source3.tsv",
    ]

    for path in train_files:
        profile_dataset(con, path)

    profile_ground_truth(
        con,
        TRAIN_DIR / "train_ground_truth.tsv"
    )

    print("\nProfiling complete.")


if __name__ == "__main__":
    main()
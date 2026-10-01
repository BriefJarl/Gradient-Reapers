from __future__ import annotations
from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"
TEST_DIR = ROOT / "student_resource" / "dataset" / "test"
OUTPUT_DIR = ROOT / "artifacts" / "normalized"

TRAIN_FILES = {
    "s1": TRAIN_DIR / "train_source1.tsv",
    "s2": TRAIN_DIR / "train_source2.tsv",
    "s3": TRAIN_DIR / "train_source3.tsv",
}

TEST_FILES = {
    "s1": TEST_DIR / "test_source1.tsv",
    "s2": TEST_DIR / "test_source2.tsv",
    "s3": TEST_DIR / "test_source3.tsv",
}

def sql_path(path: Path) -> str:
    """
    Convert Windows path to SQL-safe forward-slash path.
    """
    return str(path).replace("\\", "/")


def normalize_text_sql(column: str) -> str:
    """
    Unicode-preserving normalization.

    We deliberately remove only ASCII punctuation/control-style
    separators rather than using an ASCII-only alphanumeric class.

    This preserves:
        - Devanagari
        - accented Latin
        - Kannada
        - Bengali
        - other Unicode scripts
    """

    return f"""
        regexp_replace(
            lower(
                trim(
                    coalesce({column}, '')
                )
            ),
            '[[:punct:]]',
            ' ',
            'g'
        )
    """


def normalize_country_sql(column: str) -> str:
    return f"""
        regexp_replace(
            lower(
                trim(
                    coalesce({column}, '')
                )
            ),
            '[^a-z0-9]+',
            '',
            'g'
        )
    """



def build_source(
    con: duckdb.DuckDBPyConnection,
    source: str,
    input_path: Path,
    output_path: Path,
) -> None:

    print("\n" + "=" * 80)
    print(f"BUILDING NORMALIZED DATASET: {source.upper()}")
    print("=" * 80)

    if not input_path.exists():
        raise FileNotFoundError(
            f"Input file does not exist:\n{input_path}"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    input_sql = sql_path(input_path)
    output_sql = sql_path(output_path)

    name_norm = normalize_text_sql("business_name")
    address_norm = normalize_text_sql("business_address")
    country_norm = normalize_country_sql("country")

    query = f"""
        COPY (
            WITH base AS (
                SELECT
                    entity_id,
                    business_name,
                    business_address,
                    country,
                    {name_norm} AS name_norm_raw,
                    {address_norm} AS address_norm_raw,
                    {country_norm} AS country_norm
                FROM read_csv_auto(
                    '{input_sql}',
                    delim='\\t',
                    header=true,
                    sample_size=100000,
                    ignore_errors=false
                )
            ),

            normalized AS (
                SELECT
                    entity_id,
                    business_name,
                    business_address,
                    country,
                    regexp_replace(
                        name_norm_raw,
                        '\\s+',
                        ' ',
                        'g'
                    ) AS name_norm,
                    regexp_replace(
                        address_norm_raw,
                        '\\s+',
                        ' ',
                        'g'
                    ) AS address_norm,
                    country_norm
                FROM base
            )

            SELECT
                entity_id,
                business_name,
                business_address,
                country,
                name_norm,

                regexp_replace(
                    name_norm,
                    '[[:space:][:punct:]]',
                    '',
                    'g'
                ) AS name_compact,

                address_norm,

                regexp_replace(
                    address_norm,
                    '[[:space:][:punct:]]',
                    '',
                    'g'
                ) AS address_compact,

                country_norm,

                list_filter(
                    string_split(
                        name_norm,
                        ' '
                    ),
                    x -> trim(x) <> ''
                ) AS name_tokens,

                list_filter(
                    string_split(
                        name_norm,
                        ' '
                    ),
                    x ->
                        trim(x) <> ''
                        AND regexp_matches(x, '.*[0-9].*')
                ) AS name_numeric_tokens,

                list_filter(
                    string_split(
                        address_norm,
                        ' '
                    ),
                    x -> trim(x) <> ''
                ) AS address_tokens,

                list_filter(
                    string_split(
                        address_norm,
                        ' '
                    ),
                    x ->
                        trim(x) <> ''
                        AND regexp_matches(x, '.*[0-9].*')
                ) AS address_numeric_tokens

            FROM normalized
        )

        TO '{output_sql}'
        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    con.execute(query)

    count_query = f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
    """

    row_count = con.execute(count_query).fetchone()[0]

    print(f"Input : {input_path.name}")
    print(f"Output: {output_path}")
    print(f"Rows  : {row_count:,}")
    duplicate_ids = con.execute(
        f"""
        SELECT COUNT(*)
        FROM (
            SELECT entity_id
            FROM read_parquet('{output_sql}')
            GROUP BY entity_id
            HAVING COUNT(*) > 1
        )
        """
    ).fetchone()[0]

    empty_ids = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        WHERE entity_id IS NULL
           OR trim(entity_id) = ''
        """
    ).fetchone()[0]

    print(f"Duplicate entity IDs: {duplicate_ids:,}")
    print(f"Empty entity IDs    : {empty_ids:,}")

    if duplicate_ids != 0:
        raise ValueError(
            f"{source.upper()} contains duplicate entity IDs."
        )

    if empty_ids != 0:
        raise ValueError(
            f"{source.upper()} contains empty entity IDs."
        )

    print("Validation: PASSED")


def main() -> None:

    print("\nAMAZON ML CHALLENGE 2026")
    print("Production Normalization Pipeline")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    con = duckdb.connect()

    try:


        for source, input_path in TRAIN_FILES.items():

            output_path = (
                OUTPUT_DIR
                / f"train_{source}.parquet"
            )

            build_source(
                con,
                f"train_{source}",
                input_path,
                output_path,
            )


        for source, input_path in TEST_FILES.items():

            output_path = (
                OUTPUT_DIR
                / f"test_{source}.parquet"
            )

            build_source(
                con,
                f"test_{source}",
                input_path,
                output_path,
            )

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("NORMALIZATION PIPELINE COMPLETE")
    print("=" * 80)

    print(f"\nArtifacts created in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()

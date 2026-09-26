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

CANDIDATE_DIR = BLOCKING_DIR / "candidates"

INDEX_DIR = BLOCKING_DIR / "indexes"

CANDIDATE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# DATASETS
# ============================================================

S1_PATH = NORMALIZED_DIR / "train_s1.parquet"

TARGETS = {
    "S2": NORMALIZED_DIR / "train_s2.parquet",
    "S3": NORMALIZED_DIR / "train_s3.parquet",
}


# ============================================================
# RARE TOKEN CONFIG
# ============================================================

RARE_TOKEN_TOP_K = 2


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    """
    Convert Windows paths into DuckDB-safe paths.
    """
    return str(path).replace("\\", "/")


def rare_index_path(
    target_source: str,
    token_type: str,
) -> Path:

    return (
        INDEX_DIR
        / (
            f"train_{target_source.lower()}"
            f"_rare_{token_type}"
            f"_token_index.parquet"
        )
    )


# ============================================================
# EXACT BLOCK DEFINITIONS
# ============================================================

EXACT_BLOCKS = {

    "address": {
        "condition": """
            s1.country_norm = t.country_norm

            AND s1.address_norm = t.address_norm

            AND s1.country_norm <> ''

            AND s1.address_norm <> ''
        """,
    },

    "address_compact": {
        "condition": """
            s1.country_norm = t.country_norm

            AND s1.address_compact = t.address_compact

            AND s1.country_norm <> ''

            AND s1.address_compact <> ''
        """,
    },

    "name": {
        "condition": """
            s1.country_norm = t.country_norm

            AND s1.name_norm = t.name_norm

            AND s1.country_norm <> ''

            AND s1.name_norm <> ''
        """,
    },
}


# ============================================================
# GENERATE EXACT BLOCK
# ============================================================

def generate_exact_block(
    con: duckdb.DuckDBPyConnection,
    block_name: str,
) -> Path:

    config = EXACT_BLOCKS[block_name]

    s1 = sql_path(S1_PATH)

    output = (
        CANDIDATE_DIR
        / f"train_{block_name}_candidates.parquet"
    )

    output_sql = sql_path(output)

    print("\n" + "=" * 80)
    print(
        f"GENERATING EXACT CANDIDATES: "
        f"{block_name.upper()}"
    )
    print("=" * 80)

    for target_source, target_path in TARGETS.items():

        target = sql_path(target_path)

        source_output = (
            CANDIDATE_DIR
            / (
                f"train_{target_source.lower()}"
                f"_{block_name}"
                f"_candidates.parquet"
            )
        )

        source_output_sql = sql_path(source_output)

        print(f"\nSource: S1 -> {target_source}")

        query = f"""
            COPY (

                SELECT
                    s1.entity_id AS source1_entity_id,

                    t.entity_id AS matched_entity_id,

                    '{target_source}' AS matched_source,

                    '{block_name}' AS block_name

                FROM read_parquet('{s1}') AS s1

                INNER JOIN read_parquet('{target}') AS t

                    ON {config["condition"]}

            )

            TO '{source_output_sql}'

            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            );
        """

        con.execute(query)

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{source_output_sql}')
            """
        ).fetchone()[0]

        unique_pairs = con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source

                FROM read_parquet(
                    '{source_output_sql}'
                )
            )
            """
        ).fetchone()[0]

        print(
            f"Candidates   : {count:,}"
        )

        print(
            f"Unique pairs : {unique_pairs:,}"
        )

        print(
            f"Output       : {source_output}"
        )

    # --------------------------------------------------------
    # Combine S2 + S3
    # --------------------------------------------------------

    s2_file = sql_path(
        CANDIDATE_DIR
        / f"train_s2_{block_name}_candidates.parquet"
    )

    s3_file = sql_path(
        CANDIDATE_DIR
        / f"train_s3_{block_name}_candidates.parquet"
    )

    con.execute(
        f"""
        COPY (

            SELECT *
            FROM read_parquet('{s2_file}')

            UNION ALL

            SELECT *
            FROM read_parquet('{s3_file}')

        )

        TO '{output_sql}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    print()
    print(
        f"Combined candidates: {total:,}"
    )

    print(
        f"Output: {output}"
    )

    return output


# ============================================================
# GENERATE RARE TOKEN BLOCK
# ============================================================

def generate_rare_token_block(
    con: duckdb.DuckDBPyConnection,
    block_name: str,
) -> Path:

    if block_name == "rare_name":
        token_type = "name"
        token_column = "name_tokens"

    elif block_name == "rare_address":
        token_type = "address"
        token_column = "address_tokens"

    else:
        raise ValueError(
            f"Unsupported rare block: {block_name}"
        )

    s1 = sql_path(S1_PATH)

    output = (
        CANDIDATE_DIR
        / f"train_{block_name}_candidates.parquet"
    )

    output_sql = sql_path(output)

    print("\n" + "=" * 80)
    print(
        f"GENERATING RARE TOKEN CANDIDATES: "
        f"{block_name.upper()}"
    )
    print("=" * 80)

    print(
        f"Top rare tokens per S1 entity/source: "
        f"{RARE_TOKEN_TOP_K}"
    )

    for target_source in TARGETS:

        index_path = rare_index_path(
            target_source,
            token_type,
        )

        if not index_path.exists():
            raise FileNotFoundError(
                "\nRare-token index not found:\n"
                f"{index_path}\n\n"
                "Run build_rare_token_index.py first."
            )

        index_sql = sql_path(index_path)

        source_output = (
            CANDIDATE_DIR
            / (
                f"train_{target_source.lower()}"
                f"_{block_name}"
                f"_candidates.parquet"
            )
        )

        source_output_sql = sql_path(source_output)

        print(f"\nSource: S1 -> {target_source}")
        print(f"Index : {index_path}")

        query = f"""
            COPY (

                WITH s1_tokens AS (

                    SELECT DISTINCT

                        s1.entity_id
                            AS source1_entity_id,

                        s1.country_norm,

                        LOWER(TRIM(token))
                            AS token

                    FROM read_parquet('{s1}') AS s1

                    CROSS JOIN UNNEST(
                        s1.{token_column}
                    ) AS u(token)

                    WHERE s1.country_norm IS NOT NULL

                      AND TRIM(s1.country_norm) <> ''

                      AND token IS NOT NULL

                      AND TRIM(token) <> ''

                ),

                matched_token_keys AS (

                    SELECT DISTINCT

                        s.source1_entity_id,

                        s.country_norm,

                        s.token,

                        i.token_freq

                    FROM s1_tokens AS s

                    INNER JOIN read_parquet(
                        '{index_sql}'
                    ) AS i

                        ON s.country_norm = i.country_norm

                       AND s.token = i.token

                ),

                selected_token_keys AS (

                    SELECT

                        source1_entity_id,

                        country_norm,

                        token,

                        token_freq

                    FROM matched_token_keys

                    QUALIFY ROW_NUMBER() OVER (

                        PARTITION BY source1_entity_id

                        ORDER BY
                            token_freq ASC,
                            LENGTH(token) DESC,
                            token ASC

                    ) <= {RARE_TOKEN_TOP_K}

                )

                SELECT DISTINCT

                    s.source1_entity_id,

                    i.entity_id
                        AS matched_entity_id,

                    '{target_source}'
                        AS matched_source,

                    '{block_name}'
                        AS block_name

                FROM selected_token_keys AS s

                INNER JOIN read_parquet(
                    '{index_sql}'
                ) AS i

                    ON s.country_norm = i.country_norm

                   AND s.token = i.token

            )

            TO '{source_output_sql}'

            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            );
        """

        con.execute(query)

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet(
                '{source_output_sql}'
            )
            """
        ).fetchone()[0]

        unique_pairs = con.execute(
            f"""
            SELECT COUNT(*)
            FROM (

                SELECT DISTINCT

                    source1_entity_id,
                    matched_entity_id,
                    matched_source

                FROM read_parquet(
                    '{source_output_sql}'
                )

            )
            """
        ).fetchone()[0]

        print(
            f"Candidates   : {count:,}"
        )

        print(
            f"Unique pairs : {unique_pairs:,}"
        )

        print(
            f"Output       : {source_output}"
        )

    # --------------------------------------------------------
    # Combine S2 + S3
    # --------------------------------------------------------

    s2_file = sql_path(
        CANDIDATE_DIR
        / f"train_s2_{block_name}_candidates.parquet"
    )

    s3_file = sql_path(
        CANDIDATE_DIR
        / f"train_s3_{block_name}_candidates.parquet"
    )

    con.execute(
        f"""
        COPY (

            SELECT *
            FROM read_parquet('{s2_file}')

            UNION ALL

            SELECT *
            FROM read_parquet('{s3_file}')

        )

        TO '{output_sql}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
        """
    )

    total = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{output_sql}')
        """
    ).fetchone()[0]

    print()
    print(
        f"Combined candidates: {total:,}"
    )

    print(
        f"Output: {output}"
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Generate scalable multi-pass "
            "candidate pairs."
        )
    )

    parser.add_argument(
        "--block",
        required=True,
        choices=[
            "address",
            "address_compact",
            "name",
            "rare_name",
            "rare_address",
        ],
        help="Blocking strategy to execute.",
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("CANDIDATE GENERATION")
    print("=" * 80)

    con = duckdb.connect()

    try:

        if args.block in EXACT_BLOCKS:

            generate_exact_block(
                con,
                args.block,
            )

        else:

            generate_rare_token_block(
                con,
                args.block,
            )

    finally:
        con.close()

    print("\n" + "=" * 80)
    print("CANDIDATE GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BLOCKING_DIR = ROOT / "artifacts" / "blocking"

CANDIDATE_DIR = BLOCKING_DIR / "candidates"

UNION_DIR = BLOCKING_DIR / "union"

TEMP_DIR = BLOCKING_DIR / "tmp"

CANDIDATE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

UNION_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

TEMP_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# AVAILABLE BLOCKS
# ============================================================

BLOCK_FILES = {

    "address":
        CANDIDATE_DIR /
        "train_address_candidates.parquet",

    "address_compact":
        CANDIDATE_DIR /
        "train_address_compact_candidates.parquet",

    "name":
        CANDIDATE_DIR /
        "train_name_candidates.parquet",

    "rare_name":
        CANDIDATE_DIR /
        "train_rare_name_candidates.parquet",

    "rare_address":
        CANDIDATE_DIR /
        "train_rare_address_candidates.parquet",
}


# ============================================================
# BLOCK BIT MASKS
#
# Each blocking method gets one bit.
#
# 1  = address
# 2  = address_compact
# 4  = name
# 8  = rare_name
# 16 = rare_address
#
# Multiple blocks are combined with bitwise OR.
# ============================================================

BLOCK_BITS = {

    "address": 1,

    "address_compact": 2,

    "name": 4,

    "rare_name": 8,

    "rare_address": 16,
}


# ============================================================
# PREDEFINED BLOCK SETS
# ============================================================

BLOCK_SETS = {

    "exact": [
        "address",
        "address_compact",
        "name",
    ],

    "exact_rare_name": [
        "address",
        "address_compact",
        "name",
        "rare_name",
    ],

    "exact_rare_address": [
        "address",
        "address_compact",
        "name",
        "rare_address",
    ],

    "all": [
        "address",
        "address_compact",
        "name",
        "rare_name",
        "rare_address",
    ],
}


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    """
    Convert Windows path into DuckDB-safe path.
    """
    return str(path).replace("\\", "/")


def validate_block_files(
    block_names: list[str],
) -> None:

    missing = []

    for block_name in block_names:

        path = BLOCK_FILES[block_name]

        if not path.exists():

            missing.append(
                f"{block_name}: {path}"
            )

    if missing:

        raise FileNotFoundError(
            "\nMissing candidate files:\n"
            + "\n".join(missing)
            + "\n\n"
            "Generate the missing block first."
        )


def block_method_expression(
    block_mask: int,
) -> str:
    """
    Convert a bit-mask into a deterministic
    human-readable blocking_methods string.
    """

    expressions = []

    for block_name, bit in BLOCK_BITS.items():

        expressions.append(
            f"""
            CASE
                WHEN (blocking_mask & {bit}) <> 0
                THEN '{block_name}'
            END
            """
        )

    # We build the final method string after aggregation.
    return ", ".join(expressions)


# ============================================================
# BUILD UNION
# ============================================================

def build_union(
    con: duckdb.DuckDBPyConnection,
    block_names: list[str],
    output_name: str,
) -> Path:

    validate_block_files(
        block_names
    )

    output = (
        UNION_DIR
        / f"train_{output_name}_union.parquet"
    )

    output_sql = sql_path(output)

    # Remove an old output so that a failed run
    # cannot be mistaken for a valid result.
    if output.exists():

        output.unlink()

    print("\n" + "=" * 80)
    print("CANDIDATE UNION BUILDER")
    print("=" * 80)

    print("\nBlocks included:")

    for block_name in block_names:

        print(
            f"  - {block_name}"
            f"  [bit={BLOCK_BITS[block_name]}]"
        )

    print()
    print(
        f"Output: {output}"
    )

    # --------------------------------------------------------
    # Configure DuckDB.
    #
    # The temporary directory allows DuckDB to spill
    # intermediate data to disk instead of requiring the
    # entire operation to stay in RAM.
    # --------------------------------------------------------

    con.execute(
        f"""
        SET temp_directory = '{sql_path(TEMP_DIR)}';
        """
    )

    # Keep parallelism reasonable for a very large operation.
    con.execute(
        """
        SET threads = 8;
        """
    )

    # --------------------------------------------------------
    # Build UNION ALL.
    #
    # We attach an integer block bit instead of a string.
    # This is considerably cheaper than carrying long
    # strings through the aggregation.
    # --------------------------------------------------------

    union_parts = []

    for block_name in block_names:

        block_path = sql_path(
            BLOCK_FILES[block_name]
        )

        block_bit = BLOCK_BITS[
            block_name
        ]

        union_parts.append(
            f"""
            SELECT

                CAST(
                    source1_entity_id
                    AS VARCHAR
                ) AS source1_entity_id,

                CAST(
                    matched_entity_id
                    AS VARCHAR
                ) AS matched_entity_id,

                CAST(
                    matched_source
                    AS VARCHAR
                ) AS matched_source,

                {block_bit}
                    AS block_bit

            FROM read_parquet(
                '{block_path}'
            )
            """
        )

    raw_union_sql = (
        "\nUNION ALL\n".join(
            union_parts
        )
    )

    # --------------------------------------------------------
    # Aggregate.
    #
    # One row per unique pair.
    #
    # bit_or() preserves ALL blocking provenance.
    # --------------------------------------------------------

    query = f"""
        COPY (

            WITH raw_candidates AS (

                {raw_union_sql}

            ),

            grouped AS (

                SELECT

                    source1_entity_id,

                    matched_entity_id,

                    matched_source,

                    bit_or(block_bit)
                        AS blocking_mask

                FROM raw_candidates

                GROUP BY

                    source1_entity_id,

                    matched_entity_id,

                    matched_source

            )

            SELECT

                source1_entity_id,

                matched_entity_id,

                matched_source,

                blocking_mask,

                bit_count(
                    blocking_mask
                ) AS num_blocking_methods,

                concat_ws(
                    '|',

                    CASE
                        WHEN
                            (blocking_mask & 1) <> 0
                        THEN 'address'
                    END,

                    CASE
                        WHEN
                            (blocking_mask & 2) <> 0
                        THEN 'address_compact'
                    END,

                    CASE
                        WHEN
                            (blocking_mask & 4) <> 0
                        THEN 'name'
                    END,

                    CASE
                        WHEN
                            (blocking_mask & 8) <> 0
                        THEN 'rare_name'
                    END,

                    CASE
                        WHEN
                            (blocking_mask & 16) <> 0
                        THEN 'rare_address'
                    END

                ) AS blocking_methods

            FROM grouped

        )

        TO '{output_sql}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    print(
        "\nBuilding optimized union..."
    )

    con.execute(query)

    # --------------------------------------------------------
    # Validate output.
    # --------------------------------------------------------

    total_pairs = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{output_sql}'
        )
        """
    ).fetchone()[0]

    unique_s1 = con.execute(
        f"""
        SELECT COUNT(
            DISTINCT source1_entity_id
        )
        FROM read_parquet(
            '{output_sql}'
        )
        """
    ).fetchone()[0]

    s2_pairs = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{output_sql}'
        )
        WHERE matched_source = 'S2'
        """
    ).fetchone()[0]

    s3_pairs = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{output_sql}'
        )
        WHERE matched_source = 'S3'
        """
    ).fetchone()[0]

    # --------------------------------------------------------
    # Provenance distribution.
    # --------------------------------------------------------

    provenance_rows = con.execute(
        f"""
        SELECT
            num_blocking_methods,
            COUNT(*) AS pair_count
        FROM read_parquet(
            '{output_sql}'
        )
        GROUP BY
            num_blocking_methods
        ORDER BY
            num_blocking_methods
        """
    ).fetchall()

    print(
        "\n" + "-" * 80
    )

    print(
        "UNION SUMMARY"
    )

    print(
        "-" * 80
    )

    print(
        f"Unique candidate pairs : "
        f"{total_pairs:,}"
    )

    print(
        f"S1 entities covered    : "
        f"{unique_s1:,}"
    )

    print(
        f"S2 candidate pairs     : "
        f"{s2_pairs:,}"
    )

    print(
        f"S3 candidate pairs     : "
        f"{s3_pairs:,}"
    )

    print(
        "\nBlocking-method count:"
    )

    for count, pair_count in provenance_rows:

        print(
            f"  {count} method(s) : "
            f"{pair_count:,} pairs"
        )

    print(
        f"\nOutput: {output}"
    )

    return output


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description=(
            "Build a scalable candidate union "
            "with compact blocking provenance."
        )
    )

    parser.add_argument(
        "--set",
        dest="block_set",
        choices=sorted(
            BLOCK_SETS.keys()
        ),
        required=True,
        help="Blocking set to combine.",
    )

    args = parser.parse_args()

    block_names = BLOCK_SETS[
        args.block_set
    ]

    print("\n" + "=" * 80)
    print(
        "AMAZON ML CHALLENGE 2026"
    )
    print(
        "CANDIDATE UNION"
    )
    print("=" * 80)

    con = duckdb.connect()

    try:

        build_union(
            con=con,
            block_names=block_names,
            output_name=args.block_set,
        )

    finally:

        con.close()

    print(
        "\n" + "=" * 80
    )

    print(
        "CANDIDATE UNION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()
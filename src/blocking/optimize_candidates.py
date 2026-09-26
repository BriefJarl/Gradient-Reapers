from __future__ import annotations

from pathlib import Path
import duckdb


# ============================================================
# AMAZON ML CHALLENGE 2026
# FAST / RESUMABLE CANDIDATE OPTIMIZATION
#
# Strategy:
#   - Keep every multi-block candidate
#   - Keep name-only candidates with exact/compact name match
#   - Keep rare-address-only candidates with:
#         exact/compact name match
#         OR address token overlap >= 3
#
# IMPORTANT:
#   Actual source labels are S2 and S3.
# ============================================================


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
UNION_DIR = BLOCKING_DIR / "union"

INPUT_CANDIDATES = (
    UNION_DIR / "train_exact_rare_address_union.parquet"
)

OUTPUT_CANDIDATES = (
    UNION_DIR / "train_optimized_candidates.parquet"
)

PART_DIR = (
    UNION_DIR / "optimization_parts"
)

S1_PATH = (
    NORMALIZED_DIR / "train_s1.parquet"
)

S2_PATH = (
    NORMALIZED_DIR / "train_s2.parquet"
)

S3_PATH = (
    NORMALIZED_DIR / "train_s3.parquet"
)


# ============================================================
# SETTINGS
# ============================================================

THREADS = 8
MEMORY_LIMIT = "8GB"

# Strategy B
ADDRESS_OVERLAP_THRESHOLD = 3


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def remove_if_exists(path: Path) -> None:
    if path.exists():
        path.unlink()


def count_rows(con, path: Path) -> int:
    return con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchone()[0]


def write_part(
    con,
    part_name: str,
    query: str,
) -> Path:

    PART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    part_path = PART_DIR / part_name

    remove_if_exists(part_path)

    print()
    print("-" * 80)
    print(f"WRITING PART: {part_name}")
    print("-" * 80)

    con.execute(
        f"""
        COPY
        (
            {query}
        )
        TO '{sql_quote(part_path)}'
        (
            FORMAT PARQUET,
            COMPRESSION SNAPPY,
            ROW_GROUP_SIZE 500000
        )
        """
    )

    count = count_rows(
        con,
        part_path
    )

    print(
        f"Rows written : {count:,}"
    )

    print(
        f"File         : {part_path}"
    )

    return part_path


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("AMAZON ML CHALLENGE 2026")
    print("FAST / RESUMABLE CANDIDATE OPTIMIZATION")
    print("=" * 80)

    print()
    print(f"Threads      : {THREADS}")
    print(f"Memory limit : {MEMORY_LIMIT}")

    print()
    print(f"Input        : {INPUT_CANDIDATES}")
    print(f"Output       : {OUTPUT_CANDIDATES}")

    if not INPUT_CANDIDATES.exists():
        raise FileNotFoundError(
            f"Input candidate file not found:\n{INPUT_CANDIDATES}"
        )

    if not S1_PATH.exists():
        raise FileNotFoundError(
            f"S1 normalized file not found:\n{S1_PATH}"
        )

    if not S2_PATH.exists():
        raise FileNotFoundError(
            f"S2 normalized file not found:\n{S2_PATH}"
        )

    if not S3_PATH.exists():
        raise FileNotFoundError(
            f"S3 normalized file not found:\n{S3_PATH}"
        )

    PART_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # DuckDB
    # --------------------------------------------------------

    con = duckdb.connect()

    try:

        con.execute(
            f"SET threads = {THREADS}"
        )

        con.execute(
            f"SET memory_limit = '{MEMORY_LIMIT}'"
        )

        con.execute(
            "SET preserve_insertion_order = false"
        )

        con.execute(
            "SET enable_progress_bar = true"
        )

        temp_dir = BLOCKING_DIR / "duckdb_tmp"

        temp_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        con.execute(
            f"""
            SET temp_directory =
                '{sql_quote(temp_dir)}'
            """
        )

        # ----------------------------------------------------
        # Confirm actual source labels
        # ----------------------------------------------------

        candidate_sql = sql_quote(
            INPUT_CANDIDATES
        )

        labels = con.execute(
            f"""
            SELECT DISTINCT matched_source
            FROM read_parquet('{candidate_sql}')
            ORDER BY matched_source
            """
        ).fetchall()

        labels = [
            row[0]
            for row in labels
        ]

        print()
        print(
            f"Candidate source labels: {labels}"
        )

        expected_labels = {
            "S2",
            "S3",
        }

        if set(labels) != expected_labels:
            raise ValueError(
                "\nUnexpected source labels.\n"
                f"Expected exactly: {sorted(expected_labels)}\n"
                f"Found:            {labels}\n"
            )

        # ----------------------------------------------------
        # PART 1
        #
        # Multi-block candidates
        #
        # These were already shown to have extremely high
        # purity, so keep all of them.
        # ----------------------------------------------------

        multi_part = write_part(
            con,
            "part_multi_block.parquet",

            f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                blocking_mask,
                num_blocking_methods,
                blocking_methods

            FROM read_parquet(
                '{candidate_sql}'
            )

            WHERE num_blocking_methods >= 2
            """
        )

        # ----------------------------------------------------
        # PART 2A
        #
        # NAME-ONLY -> S2
        #
        # Require exact normalized or compact name.
        # ----------------------------------------------------

        name_s2_part = write_part(
            con,
            "part_name_s2.parquet",

            f"""
            SELECT
                c.source1_entity_id,
                c.matched_entity_id,
                c.matched_source,
                c.blocking_mask,
                c.num_blocking_methods,
                c.blocking_methods

            FROM read_parquet(
                '{candidate_sql}'
            ) c

            INNER JOIN read_parquet(
                '{sql_quote(S1_PATH)}'
            ) s1

                ON c.source1_entity_id =
                   s1.entity_id

            INNER JOIN read_parquet(
                '{sql_quote(S2_PATH)}'
            ) s2

                ON c.matched_entity_id =
                   s2.entity_id

            WHERE
                c.num_blocking_methods = 1

                AND c.blocking_methods = 'name'

                AND c.matched_source = 'S2'

                AND
                (
                    s1.name_norm =
                    s2.name_norm

                    OR

                    s1.name_compact =
                    s2.name_compact
                )
            """
        )

        # ----------------------------------------------------
        # PART 2B
        #
        # NAME-ONLY -> S3
        # ----------------------------------------------------

        name_s3_part = write_part(
            con,
            "part_name_s3.parquet",

            f"""
            SELECT
                c.source1_entity_id,
                c.matched_entity_id,
                c.matched_source,
                c.blocking_mask,
                c.num_blocking_methods,
                c.blocking_methods

            FROM read_parquet(
                '{candidate_sql}'
            ) c

            INNER JOIN read_parquet(
                '{sql_quote(S1_PATH)}'
            ) s1

                ON c.source1_entity_id =
                   s1.entity_id

            INNER JOIN read_parquet(
                '{sql_quote(S3_PATH)}'
            ) s3

                ON c.matched_entity_id =
                   s3.entity_id

            WHERE
                c.num_blocking_methods = 1

                AND c.blocking_methods = 'name'

                AND c.matched_source = 'S3'

                AND
                (
                    s1.name_norm =
                    s3.name_norm

                    OR

                    s1.name_compact =
                    s3.name_compact
                )
            """
        )

        # ----------------------------------------------------
        # PART 3A
        #
        # RARE ADDRESS -> S2
        #
        # Important optimization:
        #
        # We DO NOT build a 53M-row enriched table.
        #
        # We only load the columns needed for this query.
        # ----------------------------------------------------

        rare_s2_part = write_part(
            con,
            "part_rare_address_s2.parquet",

            f"""
            SELECT
                c.source1_entity_id,
                c.matched_entity_id,
                c.matched_source,
                c.blocking_mask,
                c.num_blocking_methods,
                c.blocking_methods

            FROM read_parquet(
                '{candidate_sql}'
            ) c

            INNER JOIN read_parquet(
                '{sql_quote(S1_PATH)}'
            ) s1

                ON c.source1_entity_id =
                   s1.entity_id

            INNER JOIN read_parquet(
                '{sql_quote(S2_PATH)}'
            ) s2

                ON c.matched_entity_id =
                   s2.entity_id

            WHERE
                c.num_blocking_methods = 1

                AND c.blocking_methods =
                    'rare_address'

                AND c.matched_source = 'S2'

                AND
                CASE

                    WHEN
                        s1.name_norm =
                        s2.name_norm

                        OR

                        s1.name_compact =
                        s2.name_compact

                    THEN TRUE

                    ELSE
                        COALESCE(
                            len(
                                list_intersect(
                                    s1.address_tokens,
                                    s2.address_tokens
                                )
                            ) >=
                            {ADDRESS_OVERLAP_THRESHOLD},
                            FALSE
                        )

                END
            """
        )

        # ----------------------------------------------------
        # PART 3B
        #
        # RARE ADDRESS -> S3
        # ----------------------------------------------------

        rare_s3_part = write_part(
            con,
            "part_rare_address_s3.parquet",

            f"""
            SELECT
                c.source1_entity_id,
                c.matched_entity_id,
                c.matched_source,
                c.blocking_mask,
                c.num_blocking_methods,
                c.blocking_methods

            FROM read_parquet(
                '{candidate_sql}'
            ) c

            INNER JOIN read_parquet(
                '{sql_quote(S1_PATH)}'
            ) s1

                ON c.source1_entity_id =
                   s1.entity_id

            INNER JOIN read_parquet(
                '{sql_quote(S3_PATH)}'
            ) s3

                ON c.matched_entity_id =
                   s3.entity_id

            WHERE
                c.num_blocking_methods = 1

                AND c.blocking_methods =
                    'rare_address'

                AND c.matched_source = 'S3'

                AND
                CASE

                    WHEN
                        s1.name_norm =
                        s3.name_norm

                        OR

                        s1.name_compact =
                        s3.name_compact

                    THEN TRUE

                    ELSE
                        COALESCE(
                            len(
                                list_intersect(
                                    s1.address_tokens,
                                    s3.address_tokens
                                )
                            ) >=
                            {ADDRESS_OVERLAP_THRESHOLD},
                            FALSE
                        )

                END
            """
        )

        # ----------------------------------------------------
        # FINAL UNION
        # ----------------------------------------------------

        print()
        print("=" * 80)
        print("BUILDING FINAL OPTIMIZED CANDIDATE FILE")
        print("=" * 80)

        remove_if_exists(
            OUTPUT_CANDIDATES
        )

        part_paths = [
            multi_part,
            name_s2_part,
            name_s3_part,
            rare_s2_part,
            rare_s3_part,
        ]

        parquet_list = ", ".join(
            f"'{sql_quote(p)}'"
            for p in part_paths
        )

        con.execute(
            f"""
            COPY
            (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    num_blocking_methods,
                    blocking_methods

                FROM read_parquet(
                    [{parquet_list}]
                )
            )

            TO '{sql_quote(OUTPUT_CANDIDATES)}'

            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 500000
            )
            """
        )

        final_count = count_rows(
            con,
            OUTPUT_CANDIDATES
        )

        print()
        print(
            f"FINAL CANDIDATES : {final_count:,}"
        )

        print()
        print(
            f"OUTPUT FILE      : {OUTPUT_CANDIDATES}"
        )

        print()
        print("=" * 80)
        print("CANDIDATE OPTIMIZATION COMPLETE")
        print("=" * 80)

    finally:

        con.close()


if __name__ == "__main__":
    main()
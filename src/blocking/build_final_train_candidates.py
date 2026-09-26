from __future__ import annotations

from pathlib import Path

import duckdb


# ============================================================
# AMAZON ML CHALLENGE 2026
# FINAL TRAIN CANDIDATE BUILDER
#
# Final candidate pool:
#
#   1. Frozen optimized base
#   2. ASCII / transliteration exact expansion
#   3. Strict rare-name expansion
#
# IMPORTANT:
# Candidate identity is ONLY:
#
#   source1_entity_id
#   matched_entity_id
#   matched_source
#
# Blocking provenance is merged separately.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

BLOCKING_DIR = ROOT / "artifacts" / "blocking"
UNION_DIR = BLOCKING_DIR / "union"
EXPERIMENT_DIR = BLOCKING_DIR / "experiments"
TMP_DIR = BLOCKING_DIR / "duckdb_tmp"

BASE = UNION_DIR / "train_optimized_candidates.parquet"

ASCII = (
    EXPERIMENT_DIR
    / "train_ascii_exact_candidates.parquet"
)

RARE_NAME = (
    EXPERIMENT_DIR
    / "train_rare_name_strict_candidates.parquet"
)

OUTPUT = (
    UNION_DIR
    / "train_final_candidates.parquet"
)


# ============================================================
# HELPERS
# ============================================================

def sql_path(path: Path) -> str:
    return (
        str(path.resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )


def check_inputs() -> None:

    required = [
        BASE,
        ASCII,
        RARE_NAME,
    ]

    missing = [
        str(path)
        for path in required
        if not path.exists()
    ]

    if missing:
        raise FileNotFoundError(
            "Missing required candidate file(s):\n"
            + "\n".join(
                f"  - {x}"
                for x in missing
            )
        )


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 90)
    print("AMAZON ML CHALLENGE 2026")
    print("FINAL TRAIN CANDIDATE BUILDER")
    print("=" * 90)

    check_inputs()

    UNION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    TMP_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print()
    print(f"BASE      : {BASE}")
    print(f"ASCII     : {ASCII}")
    print(f"RARE NAME : {RARE_NAME}")
    print(f"OUTPUT    : {OUTPUT}")

    con = duckdb.connect()

    try:

        # ----------------------------------------------------
        # DuckDB configuration
        # ----------------------------------------------------

        con.execute("SET threads=8")
        con.execute("SET memory_limit='8GB'")
        con.execute(
            "SET preserve_insertion_order=false"
        )

        con.execute(
            f"SET temp_directory='{sql_path(TMP_DIR)}'"
        )

        con.execute(
            "SET enable_progress_bar=true"
        )

        base = sql_path(BASE)
        ascii_path = sql_path(ASCII)
        rare_path = sql_path(RARE_NAME)
        output = sql_path(OUTPUT)

        # ----------------------------------------------------
        # 1. Inspect schemas
        # ----------------------------------------------------

        print()
        print("[1/8] Inspecting candidate schemas...")

        for label, path in [
            ("BASE", base),
            ("ASCII", ascii_path),
            ("RARE_NAME", rare_path),
        ]:

            print(f"\n--- {label} ---")

            rows = con.execute(
                f"""
                DESCRIBE
                SELECT *
                FROM read_parquet('{path}')
                """
            ).fetchall()

            for row in rows:
                print(
                    f"  {row[0]:25s} {row[1]}"
                )

        # ----------------------------------------------------
        # 2. Count inputs
        # ----------------------------------------------------

        print()
        print("[2/8] Counting input candidate pools...")

        base_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{base}')
            """
        ).fetchone()[0]

        ascii_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{ascii_path}')
            """
        ).fetchone()[0]

        rare_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{rare_path}')
            """
        ).fetchone()[0]

        print(f"BASE      : {base_count:,}")
        print(f"ASCII     : {ascii_count:,}")
        print(f"RARE NAME : {rare_count:,}")

        # ----------------------------------------------------
        # 3. Validate frozen base itself
        # ----------------------------------------------------

        print()
        print("[3/8] Validating frozen base...")

        base_duplicate_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    COUNT(*) AS n
                FROM read_parquet('{base}')
                GROUP BY
                    source1_entity_id,
                    matched_entity_id,
                    matched_source
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        base_null_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{base}')
            WHERE source1_entity_id IS NULL
               OR matched_entity_id IS NULL
               OR matched_source IS NULL
            """
        ).fetchone()[0]

        print(
            f"Base duplicate pair groups : "
            f"{base_duplicate_count:,}"
        )

        print(
            f"Base NULL critical IDs     : "
            f"{base_null_count:,}"
        )

        if base_duplicate_count != 0:
            raise RuntimeError(
                "FROZEN BASE ALREADY CONTAINS DUPLICATE "
                "CANDIDATE PAIRS."
            )

        if base_null_count != 0:
            raise RuntimeError(
                "FROZEN BASE CONTAINS NULL CRITICAL IDs."
            )

        # ----------------------------------------------------
        # 4. Build raw expansion relation
        #
        # DO NOT use SELECT DISTINCT * here.
        #
        # We intentionally retain every discovery because
        # provenance must be merged by candidate identity.
        # ----------------------------------------------------

        print()
        print("[4/8] Building raw expansion relation...")

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW expansion_raw AS

            SELECT
                CAST(source1_entity_id AS VARCHAR)
                    AS source1_entity_id,

                CAST(matched_entity_id AS VARCHAR)
                    AS matched_entity_id,

                CASE
                    WHEN UPPER(
                        CAST(matched_source AS VARCHAR)
                    ) IN (
                        'S2',
                        'SOURCE2',
                        'SOURCE_2',
                        '2'
                    )
                    THEN 'S2'

                    WHEN UPPER(
                        CAST(matched_source AS VARCHAR)
                    ) IN (
                        'S3',
                        'SOURCE3',
                        'SOURCE_3',
                        '3'
                    )
                    THEN 'S3'

                    ELSE UPPER(
                        CAST(matched_source AS VARCHAR)
                    )
                END AS matched_source,

                CAST(blocking_mask AS INTEGER)
                    AS blocking_mask,

                CAST(num_blocking_methods AS TINYINT)
                    AS num_blocking_methods,

                CAST(blocking_methods AS VARCHAR)
                    AS blocking_methods

            FROM read_parquet('{ascii_path}')

            WHERE source1_entity_id IS NOT NULL
              AND matched_entity_id IS NOT NULL

            UNION ALL

            SELECT
                CAST(source1_entity_id AS VARCHAR)
                    AS source1_entity_id,

                CAST(matched_entity_id AS VARCHAR)
                    AS matched_entity_id,

                CASE
                    WHEN UPPER(
                        CAST(matched_source AS VARCHAR)
                    ) IN (
                        'S2',
                        'SOURCE2',
                        'SOURCE_2',
                        '2'
                    )
                    THEN 'S2'

                    WHEN UPPER(
                        CAST(matched_source AS VARCHAR)
                    ) IN (
                        'S3',
                        'SOURCE3',
                        'SOURCE_3',
                        '3'
                    )
                    THEN 'S3'

                    ELSE UPPER(
                        CAST(matched_source AS VARCHAR)
                    )
                END AS matched_source,

                CAST(blocking_mask AS INTEGER)
                    AS blocking_mask,

                CAST(num_blocking_methods AS TINYINT)
                    AS num_blocking_methods,

                CAST(blocking_methods AS VARCHAR)
                    AS blocking_methods

            FROM read_parquet('{rare_path}')

            WHERE source1_entity_id IS NOT NULL
              AND matched_entity_id IS NOT NULL
            """
        )

        raw_count = con.execute(
            """
            SELECT COUNT(*)
            FROM expansion_raw
            """
        ).fetchone()[0]

        print(
            f"Raw expansion rows : "
            f"{raw_count:,}"
        )

        # ----------------------------------------------------
        # 5. CRITICAL:
        # Deduplicate by candidate identity.
        #
        # Identity:
        #
        #   S1 + target + source
        #
        # Provenance:
        #
        #   BIT_OR(mask)
        #   merged method list
        #
        # This is the actual fix for your 47,041 duplicates.
        # ----------------------------------------------------

        print()
        print(
            "[5/8] Deduplicating expansion "
            "by candidate identity..."
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW expansion_unique AS

            SELECT

                source1_entity_id,

                matched_entity_id,

                matched_source,

                BIT_OR(blocking_mask)
                    AS blocking_mask,

                COUNT(
                    DISTINCT blocking_methods
                )::TINYINT
                    AS num_blocking_methods,

                STRING_AGG(
                    DISTINCT blocking_methods,
                    '|'
                ) AS blocking_methods

            FROM expansion_raw

            GROUP BY
                source1_entity_id,
                matched_entity_id,
                matched_source
            """
        )

        expansion_unique_count = con.execute(
            """
            SELECT COUNT(*)
            FROM expansion_unique
            """
        ).fetchone()[0]

        print(
            f"Unique expansion pairs : "
            f"{expansion_unique_count:,}"
        )

        # ----------------------------------------------------
        # 6. Remove anything already in base
        # ----------------------------------------------------

        print()
        print(
            "[6/8] Removing candidates already "
            "present in frozen base..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW new_expansion AS

            SELECT
                e.source1_entity_id,
                e.matched_entity_id,
                e.matched_source,
                e.blocking_mask,
                e.num_blocking_methods,
                e.blocking_methods

            FROM expansion_unique e

            ANTI JOIN
            (
                SELECT
                    CAST(source1_entity_id AS VARCHAR)
                        AS source1_entity_id,

                    CAST(matched_entity_id AS VARCHAR)
                        AS matched_entity_id,

                    CASE
                        WHEN UPPER(
                            CAST(matched_source AS VARCHAR)
                        ) IN (
                            'S2',
                            'SOURCE2',
                            'SOURCE_2',
                            '2'
                        )
                        THEN 'S2'

                        WHEN UPPER(
                            CAST(matched_source AS VARCHAR)
                        ) IN (
                            'S3',
                            'SOURCE3',
                            'SOURCE_3',
                            '3'
                        )
                        THEN 'S3'

                        ELSE UPPER(
                            CAST(matched_source AS VARCHAR)
                        )
                    END AS matched_source

                FROM read_parquet('{base}')
            ) b

            ON e.source1_entity_id =
               b.source1_entity_id

           AND e.matched_entity_id =
               b.matched_entity_id

           AND e.matched_source =
               b.matched_source
            """
        )

        new_count = con.execute(
            """
            SELECT COUNT(*)
            FROM new_expansion
            """
        ).fetchone()[0]

        print(
            f"NEW candidates over base : "
            f"{new_count:,}"
        )

        # ----------------------------------------------------
        # Expected final count
        # ----------------------------------------------------

        expected_final_count = (
            base_count +
            new_count
        )

        # ----------------------------------------------------
        # 7. Write final candidate parquet
        # ----------------------------------------------------

        print()
        print(
            "[7/8] Writing final candidate parquet..."
        )

        if OUTPUT.exists():
            OUTPUT.unlink()

        con.execute(
            f"""
            COPY
            (
                SELECT
                    CAST(source1_entity_id AS VARCHAR)
                        AS source1_entity_id,

                    CAST(matched_entity_id AS VARCHAR)
                        AS matched_entity_id,

                    CASE
                        WHEN UPPER(
                            CAST(matched_source AS VARCHAR)
                        ) IN (
                            'S2',
                            'SOURCE2',
                            'SOURCE_2',
                            '2'
                        )
                        THEN 'S2'

                        WHEN UPPER(
                            CAST(matched_source AS VARCHAR)
                        ) IN (
                            'S3',
                            'SOURCE3',
                            'SOURCE_3',
                            '3'
                        )
                        THEN 'S3'

                        ELSE UPPER(
                            CAST(matched_source AS VARCHAR)
                        )
                    END AS matched_source,

                    CAST(blocking_mask AS INTEGER)
                        AS blocking_mask,

                    CAST(num_blocking_methods AS TINYINT)
                        AS num_blocking_methods,

                    CAST(blocking_methods AS VARCHAR)
                        AS blocking_methods

                FROM read_parquet('{base}')

                UNION ALL

                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    num_blocking_methods,
                    blocking_methods

                FROM new_expansion
            )

            TO '{output}'

            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ----------------------------------------------------
        # 8. Final validation
        # ----------------------------------------------------

        print()
        print(
            "[8/8] Running final validation..."
        )

        final_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output}')
            """
        ).fetchone()[0]

        duplicate_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source

                FROM read_parquet('{output}')

                GROUP BY
                    source1_entity_id,
                    matched_entity_id,
                    matched_source

                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        null_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output}')
            WHERE source1_entity_id IS NULL
               OR matched_entity_id IS NULL
               OR matched_source IS NULL
            """
        ).fetchone()[0]

        invalid_source_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output}')
            WHERE matched_source NOT IN ('S2', 'S3')
            """
        ).fetchone()[0]

        print()
        print(
            f"Base candidates          : "
            f"{base_count:,}"
        )

        print(
            f"Raw expansion rows       : "
            f"{raw_count:,}"
        )

        print(
            f"Unique expansion pairs   : "
            f"{expansion_unique_count:,}"
        )

        print(
            f"NEW expansion candidates : "
            f"{new_count:,}"
        )

        print(
            f"Expected final count     : "
            f"{expected_final_count:,}"
        )

        print(
            f"Actual final count       : "
            f"{final_count:,}"
        )

        print(
            f"Duplicate pair groups    : "
            f"{duplicate_count:,}"
        )

        print(
            f"Null critical IDs        : "
            f"{null_count:,}"
        )

        print(
            f"Invalid source labels    : "
            f"{invalid_source_count:,}"
        )

        # ----------------------------------------------------
        # Source distribution
        # ----------------------------------------------------

        print()
        print("FINAL SOURCE DISTRIBUTION")

        source_rows = con.execute(
            f"""
            SELECT
                matched_source,
                COUNT(*) AS n

            FROM read_parquet('{output}')

            GROUP BY matched_source

            ORDER BY matched_source
            """
        ).fetchall()

        for source, count in source_rows:
            print(
                f"  {source}: {count:,}"
            )

        # ----------------------------------------------------
        # Hard assertions
        # ----------------------------------------------------

        if expansion_unique_count > raw_count:
            raise RuntimeError(
                "UNIQUE EXPANSION COUNT CANNOT EXCEED "
                "RAW EXPANSION COUNT."
            )

        if final_count != expected_final_count:
            raise RuntimeError(
                "FINAL COUNT MISMATCH: "
                f"expected {expected_final_count:,}, "
                f"got {final_count:,}"
            )

        if duplicate_count != 0:
            raise RuntimeError(
                "DUPLICATE CANDIDATE PAIRS DETECTED: "
                f"{duplicate_count:,}"
            )

        if null_count != 0:
            raise RuntimeError(
                "NULL CRITICAL IDs DETECTED: "
                f"{null_count:,}"
            )

        if invalid_source_count != 0:
            raise RuntimeError(
                "INVALID MATCHED_SOURCE VALUES DETECTED: "
                f"{invalid_source_count:,}"
            )

        # ----------------------------------------------------
        # SUCCESS
        # ----------------------------------------------------

        print()
        print("=" * 90)
        print(
            "FINAL TRAIN CANDIDATE BUILD PASSED"
        )
        print("=" * 90)

        print(
            f"Output : {OUTPUT}"
        )

        print(
            f"Rows   : {final_count:,}"
        )

        print("=" * 90)

    finally:
        con.close()


if __name__ == "__main__":
    main()
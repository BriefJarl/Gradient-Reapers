from __future__ import annotations

from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parents[2]

BLOCKING_DIR = ROOT / "artifacts" / "blocking"
UNION_DIR = BLOCKING_DIR / "union"
CANDIDATE_DIR = BLOCKING_DIR / "candidates"

BASE = UNION_DIR / "train_optimized_candidates.parquet"
RARE_NAME = CANDIDATE_DIR / "train_rare_name_candidates.parquet"

OUTPUT = UNION_DIR / "train_optimized_plus_rare_name.parquet"

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def count_rows(con, path: Path) -> int:
    return con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchone()[0]


def get_columns(con, path: Path) -> list[str]:
    rows = con.execute(
        f"""
        DESCRIBE SELECT *
        FROM read_parquet('{sql_quote(path)}')
        """
    ).fetchall()

    return [row[0] for row in rows]


def find_column(columns: list[str], candidates: list[str]) -> str | None:
    lookup = {c.lower(): c for c in columns}

    for candidate in candidates:
        if candidate.lower() in lookup:
            return lookup[candidate.lower()]

    return None


def main() -> None:

    print("=" * 80)
    print("RARE-NAME UNION EXPERIMENT")
    print("=" * 80)

    if not BASE.exists():
        raise FileNotFoundError(
            f"Base candidate file not found:\n{BASE}"
        )

    if not RARE_NAME.exists():
        raise FileNotFoundError(
            f"Rare-name candidate file not found:\n{RARE_NAME}"
        )

    print()
    print(f"BASE      : {BASE}")
    print(f"RARE NAME : {RARE_NAME}")
    print(f"OUTPUT    : {OUTPUT}")

    con = duckdb.connect()

    try:

        # ------------------------------------------------------------
        # DuckDB configuration
        # ------------------------------------------------------------

        con.execute(f"SET threads = {THREADS}")
        con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = true")

        temp_dir = BLOCKING_DIR / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        con.execute(
            f"SET temp_directory = '{sql_quote(temp_dir)}'"
        )

        # ------------------------------------------------------------
        # Inspect schemas
        # ------------------------------------------------------------

        base_columns = get_columns(con, BASE)
        rare_columns = get_columns(con, RARE_NAME)

        print()
        print("BASE COLUMNS:")
        print(", ".join(base_columns))

        print()
        print("RARE-NAME COLUMNS:")
        print(", ".join(rare_columns))

        # ------------------------------------------------------------
        # Resolve important columns
        # ------------------------------------------------------------

        s1_col = find_column(
            rare_columns,
            [
                "source1_entity_id",
                "entity_id",
                "source1_id",
            ],
        )

        matched_col = find_column(
            rare_columns,
            [
                "matched_entity_id",
                "entity_id_source2",
                "matched_id",
            ],
        )

        source_col = find_column(
            rare_columns,
            [
                "matched_source",
                "source",
                "matched_source_name",
            ],
        )

        block_col = find_column(
            rare_columns,
            [
                "block_name",
                "blocking_method",
                "block",
            ],
        )

        if s1_col is None:
            raise ValueError(
                "Could not identify Source-1 entity ID column "
                f"in rare-name file.\nColumns: {rare_columns}"
            )

        if matched_col is None:
            raise ValueError(
                "Could not identify matched entity ID column "
                f"in rare-name file.\nColumns: {rare_columns}"
            )

        if source_col is None:
            raise ValueError(
                "Could not identify matched source column "
                f"in rare-name file.\nColumns: {rare_columns}"
            )

        if block_col is None:
            raise ValueError(
                "Could not identify blocking method column "
                f"in rare-name file.\nColumns: {rare_columns}"
            )

        print()
        print("Resolved rare-name schema:")
        print(f"  S1 ID       : {s1_col}")
        print(f"  Matched ID  : {matched_col}")
        print(f"  Source      : {source_col}")
        print(f"  Block       : {block_col}")

        # ------------------------------------------------------------
        # Counts
        # ------------------------------------------------------------

        base_count = count_rows(con, BASE)
        rare_count = count_rows(con, RARE_NAME)

        print()
        print(f"Base candidates      : {base_count:,}")
        print(f"Rare-name candidates : {rare_count:,}")

        # ------------------------------------------------------------
        # Remove old experiment output if present
        # ------------------------------------------------------------

        if OUTPUT.exists():
            OUTPUT.unlink()

        # ------------------------------------------------------------
        # Build UNION
        #
        # Existing optimized candidates already have:
        #   blocking_mask
        #   num_blocking_methods
        #   blocking_methods
        #
        # Rare-name candidates are converted to:
        #   blocking_mask = 8
        #   num_blocking_methods = 1
        #   blocking_methods = rare_name
        # ------------------------------------------------------------

        print()
        print("-" * 80)
        print("BUILDING RARE-NAME UNION")
        print("-" * 80)

        base_sql = sql_quote(BASE)
        rare_sql = sql_quote(RARE_NAME)
        output_sql = sql_quote(OUTPUT)

        con.execute(
            f"""
            COPY
            (
                WITH combined AS
                (
                    SELECT
                        source1_entity_id,
                        matched_entity_id,
                        matched_source,
                        blocking_mask,
                        num_blocking_methods,
                        blocking_methods
                    FROM read_parquet('{base_sql}')

                    UNION ALL

                    SELECT
                        {s1_col} AS source1_entity_id,
                        {matched_col} AS matched_entity_id,
                        {source_col} AS matched_source,

                        8 AS blocking_mask,

                        1 AS num_blocking_methods,

                        'rare_name' AS blocking_methods

                    FROM read_parquet('{rare_sql}')
                ),

                deduplicated AS
                (
                    SELECT
                        source1_entity_id,
                        matched_entity_id,
                        matched_source,

                        BIT_OR(blocking_mask) AS blocking_mask,

                        COUNT(DISTINCT blocking_methods)
                            AS num_blocking_methods,

                        STRING_AGG(
                            DISTINCT blocking_methods,
                            '|'
                            ORDER BY blocking_methods
                        ) AS blocking_methods

                    FROM combined

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
                    CAST(num_blocking_methods AS TINYINT)
                        AS num_blocking_methods,
                    blocking_methods

                FROM deduplicated
            )

            TO '{output_sql}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 500000
            )
            """
        )

        # ------------------------------------------------------------
        # Final statistics
        # ------------------------------------------------------------

        final_count = count_rows(con, OUTPUT)

        print()
        print("=" * 80)
        print("RARE-NAME UNION COMPLETE")
        print("=" * 80)

        print()
        print(f"Original candidates : {base_count:,}")
        print(f"Rare-name input     : {rare_count:,}")
        print(f"Final unique        : {final_count:,}")
        print(
            f"Additional unique   : "
            f"{final_count - base_count:,}"
        )

        print()
        print(f"OUTPUT:")
        print(OUTPUT)

        print()
        print("=" * 80)
        print("SUCCESS")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
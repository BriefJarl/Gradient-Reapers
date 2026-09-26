from __future__ import annotations

from pathlib import Path
import duckdb


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

BLOCKING_DIR = ROOT / "artifacts" / "blocking"
UNION_DIR = BLOCKING_DIR / "union"
CANDIDATE_DIR = BLOCKING_DIR / "candidates"

BASE = UNION_DIR / "train_optimized_candidates.parquet"
RARE_NAME = CANDIDATE_DIR / "train_rare_name_candidates.parquet"
GROUND_TRUTH = BLOCKING_DIR / "ground_truth_pairs.parquet"

THREADS = 8
MEMORY_LIMIT = "8GB"


# ============================================================
# HELPERS
# ============================================================

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


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    print("=" * 80)
    print("RARE-NAME INCREMENTAL BLOCKING ANALYSIS")
    print("=" * 80)

    # --------------------------------------------------------
    # Validate files
    # --------------------------------------------------------

    for path in [BASE, RARE_NAME, GROUND_TRUTH]:
        if not path.exists():
            raise FileNotFoundError(
                f"\nRequired file not found:\n{path}"
            )

    print()
    print(f"BASE         : {BASE}")
    print(f"RARE NAME    : {RARE_NAME}")
    print(f"GROUND TRUTH : {GROUND_TRUTH}")

    # --------------------------------------------------------
    # DuckDB
    # --------------------------------------------------------

    con = duckdb.connect()

    try:

        con.execute(f"SET threads = {THREADS}")
        con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
        con.execute("SET preserve_insertion_order = false")
        con.execute("SET enable_progress_bar = true")

        temp_dir = BLOCKING_DIR / "duckdb_tmp"
        temp_dir.mkdir(parents=True, exist_ok=True)

        con.execute(
            f"SET temp_directory = '{sql_quote(temp_dir)}'"
        )

        base_sql = sql_quote(BASE)
        rare_sql = sql_quote(RARE_NAME)
        gt_sql = sql_quote(GROUND_TRUTH)

        # ----------------------------------------------------
        # Basic counts
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("BASIC COUNTS")
        print("-" * 80)

        base_count = count_rows(con, BASE)
        rare_count = count_rows(con, RARE_NAME)
        gt_count = count_rows(con, GROUND_TRUTH)

        print(f"Base candidates       : {base_count:,}")
        print(f"Rare-name candidates  : {rare_count:,}")
        print(f"Ground-truth pairs    : {gt_count:,}")

        # ----------------------------------------------------
        # Inspect schemas
        # ----------------------------------------------------

        base_columns = get_columns(con, BASE)
        rare_columns = get_columns(con, RARE_NAME)
        gt_columns = get_columns(con, GROUND_TRUTH)

        print()
        print("BASE COLUMNS:")
        print(", ".join(base_columns))

        print()
        print("RARE-NAME COLUMNS:")
        print(", ".join(rare_columns))

        print()
        print("GROUND-TRUTH COLUMNS:")
        print(", ".join(gt_columns))

        # ----------------------------------------------------
        # Check duplicate pairs in rare-name candidates
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("CHECKING RARE-NAME DUPLICATES")
        print("-" * 80)

        rare_distinct_pairs = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT DISTINCT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source
                FROM read_parquet('{rare_sql}')
            )
            """
        ).fetchone()[0]

        rare_duplicate_rows = rare_count - rare_distinct_pairs

        print(f"Rare-name rows          : {rare_count:,}")
        print(f"Distinct rare-name     : {rare_distinct_pairs:,}")
        print(f"Duplicate rows         : {rare_duplicate_rows:,}")

        # ----------------------------------------------------
        # True pairs recovered by rare-name
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("RARE-NAME TRUE-PAIR RECOVERY")
        print("-" * 80)

        rare_true_pairs = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT DISTINCT
                    r.source1_entity_id,
                    r.matched_entity_id,
                    r.matched_source

                FROM read_parquet('{rare_sql}') r

                INNER JOIN read_parquet('{gt_sql}') g
                    ON r.source1_entity_id = g.source1_entity_id
                   AND r.matched_entity_id = g.matched_entity_id
                   AND r.matched_source = g.matched_source
            )
            """
        ).fetchone()[0]

        print(
            f"True pairs recovered by rare-name : "
            f"{rare_true_pairs:,}"
        )

        rare_recall = (
            rare_true_pairs / gt_count
            if gt_count
            else 0.0
        )

        print(
            f"Rare-name standalone recall       : "
            f"{rare_recall * 100:.6f}%"
        )

        # ----------------------------------------------------
        # TRUE pairs already recovered by BASE
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("BASE TRUE-PAIR RECOVERY")
        print("-" * 80)

        base_true_pairs = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT DISTINCT
                    b.source1_entity_id,
                    b.matched_entity_id,
                    b.matched_source

                FROM read_parquet('{base_sql}') b

                INNER JOIN read_parquet('{gt_sql}') g
                    ON b.source1_entity_id = g.source1_entity_id
                   AND b.matched_entity_id = g.matched_entity_id
                   AND b.matched_source = g.matched_source
            )
            """
        ).fetchone()[0]

        print(
            f"Base true pairs : {base_true_pairs:,}"
        )

        # ----------------------------------------------------
        # INCREMENTAL TRUE PAIRS
        #
        # This is the critical calculation.
        #
        # Rare-name true pairs that are NOT already in BASE.
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("INCREMENTAL TRUE-PAIR RECOVERY")
        print("-" * 80)

        incremental_true_pairs = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT DISTINCT
                    r.source1_entity_id,
                    r.matched_entity_id,
                    r.matched_source

                FROM read_parquet('{rare_sql}') r

                INNER JOIN read_parquet('{gt_sql}') g
                    ON r.source1_entity_id = g.source1_entity_id
                   AND r.matched_entity_id = g.matched_entity_id
                   AND r.matched_source = g.matched_source

                ANTI JOIN
                (
                    SELECT DISTINCT
                        source1_entity_id,
                        matched_entity_id,
                        matched_source
                    FROM read_parquet('{base_sql}')
                ) b

                    ON r.source1_entity_id = b.source1_entity_id
                   AND r.matched_entity_id = b.matched_entity_id
                   AND r.matched_source = b.matched_source
            )
            """
        ).fetchone()[0]

        print(
            f"NEW true pairs from rare-name : "
            f"{incremental_true_pairs:,}"
        )

        incremental_recall = (
            incremental_true_pairs / gt_count
            if gt_count
            else 0.0
        )

        print(
            f"Incremental recall gain       : "
            f"{incremental_recall * 100:.6f}%"
        )

        # ----------------------------------------------------
        # NEW CANDIDATE ESTIMATE
        #
        # Distinct rare-name candidates not in BASE.
        # We do this separately from the truth calculation.
        # ----------------------------------------------------

        print()
        print("-" * 80)
        print("INCREMENTAL CANDIDATE VOLUME")
        print("-" * 80)

        new_candidate_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT DISTINCT
                    r.source1_entity_id,
                    r.matched_entity_id,
                    r.matched_source

                FROM read_parquet('{rare_sql}') r

                ANTI JOIN
                (
                    SELECT DISTINCT
                        source1_entity_id,
                        matched_entity_id,
                        matched_source
                    FROM read_parquet('{base_sql}')
                ) b

                    ON r.source1_entity_id = b.source1_entity_id
                   AND r.matched_entity_id = b.matched_entity_id
                   AND r.matched_source = b.matched_source
            )
            """
        ).fetchone()[0]

        print(
            f"New unique candidates : "
            f"{new_candidate_count:,}"
        )

        # ----------------------------------------------------
        # Incremental purity
        # ----------------------------------------------------

        incremental_purity = (
            incremental_true_pairs / new_candidate_count
            if new_candidate_count
            else 0.0
        )

        print(
            f"Incremental purity     : "
            f"{incremental_purity * 100:.6f}%"
        )

        # ----------------------------------------------------
        # Projected combined statistics
        # ----------------------------------------------------

        projected_candidates = (
            base_count + new_candidate_count
        )

        projected_true_pairs = (
            base_true_pairs + incremental_true_pairs
        )

        projected_recall = (
            projected_true_pairs / gt_count
            if gt_count
            else 0.0
        )

        projected_purity = (
            projected_true_pairs / projected_candidates
            if projected_candidates
            else 0.0
        )

        print()
        print("=" * 80)
        print("PROJECTED COMBINED BLOCKING")
        print("=" * 80)

        print()
        print(
            f"Projected candidates : "
            f"{projected_candidates:,}"
        )

        print(
            f"Projected true pairs : "
            f"{projected_true_pairs:,}"
        )

        print(
            f"Projected recall     : "
            f"{projected_recall * 100:.6f}%"
        )

        print(
            f"Projected purity     : "
            f"{projected_purity * 100:.6f}%"
        )

        print()
        print("=" * 80)
        print("RARE-NAME ANALYSIS COMPLETE")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
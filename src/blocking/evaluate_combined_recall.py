from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

BLOCKING = ROOT / "artifacts" / "blocking"
UNION_DIR = BLOCKING / "union"
EXPERIMENTS = BLOCKING / "experiments"
TMP = BLOCKING / "duckdb_tmp"

BASE = UNION_DIR / "train_optimized_candidates.parquet"
ASCII = EXPERIMENTS / "train_ascii_exact_candidates.parquet"
RARE_NAME = EXPERIMENTS / "train_rare_name_strict_candidates.parquet"
GT = BLOCKING / "ground_truth_pairs.parquet"


def sql_path(path: Path) -> str:
    return (
        str(path.resolve())
        .replace("\\", "/")
        .replace("'", "''")
    )


def configure(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads=8")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")

    TMP.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"SET temp_directory='{sql_path(TMP)}'"
    )

    con.execute(
        "SET enable_progress_bar=true"
    )


def count(con, sql: str) -> int:
    return int(con.execute(sql).fetchone()[0])


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        default=str(BASE),
    )

    parser.add_argument(
        "--ascii",
        default=str(ASCII),
    )

    parser.add_argument(
        "--rare-name",
        default=str(RARE_NAME),
    )

    parser.add_argument(
        "--ground-truth",
        default=str(GT),
    )

    args = parser.parse_args()

    base = Path(args.base)
    ascii_file = Path(args.ascii)
    rare_file = Path(args.rare_name)
    gt = Path(args.ground_truth)

    for path in [base, ascii_file, rare_file, gt]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    print("=" * 90)
    print("COMBINED RECALL EVALUATION")
    print("=" * 90)

    print(f"Base       : {base}")
    print(f"ASCII      : {ascii_file}")
    print(f"Rare-name  : {rare_file}")
    print(f"Groundtruth: {gt}")

    con = duckdb.connect()

    try:
        configure(con)

        b = sql_path(base)
        a = sql_path(ascii_file)
        r = sql_path(rare_file)
        g = sql_path(gt)

        # -----------------------------------------------------
        # Counts
        # -----------------------------------------------------

        base_count = count(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{b}')
            """
        )

        ascii_count = count(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{a}')
            """
        )

        rare_count = count(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{r}')
            """
        )

        total_true = count(
            con,
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{g}')
            """
        )

        print()
        print(f"Base candidates : {base_count:,}")
        print(f"ASCII candidates: {ascii_count:,}")
        print(f"Rare candidates : {rare_count:,}")
        print(f"True pairs      : {total_true:,}")

        # -----------------------------------------------------
        # Build only the NEW candidate pool.
        #
        # This avoids materializing:
        # base + ascii + rare
        # with duplicate rows.
        # -----------------------------------------------------

        print()
        print("[1/4] Finding candidates not already in base...")

        con.execute(
            """
            DROP TABLE IF EXISTS expansion_candidates
            """
        )

        con.execute(
            f"""
            CREATE TEMP TABLE expansion_candidates AS

            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source
            FROM read_parquet('{a}')

            UNION

            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source
            FROM read_parquet('{r}')
            """
        )

        expansion_count = count(
            con,
            """
            SELECT COUNT(*)
            FROM expansion_candidates
            """
        )

        print(
            f"Combined ASCII + rare-name candidates: "
            f"{expansion_count:,}"
        )

        print()
        print("[2/4] Removing candidates already in base...")

        con.execute(
            """
            DROP TABLE IF EXISTS new_candidates
            """
        )

        con.execute(
            f"""
            CREATE TEMP TABLE new_candidates AS

            SELECT e.*

            FROM expansion_candidates e

            ANTI JOIN read_parquet('{b}') base

              ON e.source1_entity_id =
                 base.source1_entity_id

             AND e.matched_entity_id =
                 base.matched_entity_id

             AND e.matched_source =
                 base.matched_source
            """
        )

        new_count = count(
            con,
            """
            SELECT COUNT(*)
            FROM new_candidates
            """
        )

        print(
            f"NEW candidates over base: {new_count:,}"
        )

        print()
        print("[3/4] Measuring newly recovered true pairs...")

        new_true = count(
            con,
            f"""
            SELECT COUNT(*)

            FROM new_candidates c

            INNER JOIN read_parquet('{g}') gt

              ON c.source1_entity_id =
                 gt.source1_entity_id

             AND c.matched_entity_id =
                 gt.matched_entity_id

             AND c.matched_source =
                 gt.matched_source
            """
        )

        print(
            f"NEW true pairs: {new_true:,}"
        )

        print()
        print("[4/4] Measuring base recall...")

        base_true = count(
            con,
            f"""
            SELECT COUNT(*)

            FROM read_parquet('{b}') c

            INNER JOIN read_parquet('{g}') gt

              ON c.source1_entity_id =
                 gt.source1_entity_id

             AND c.matched_entity_id =
                 gt.matched_entity_id

             AND c.matched_source =
                 gt.matched_source
            """
        )

        combined_candidates = base_count + new_count
        combined_true = base_true + new_true

        incremental_recall = (
            new_true / total_true
            if total_true
            else 0.0
        )

        combined_recall = (
            combined_true / total_true
            if total_true
            else 0.0
        )

        incremental_purity = (
            new_true / new_count
            if new_count
            else 0.0
        )

        combined_purity = (
            combined_true / combined_candidates
            if combined_candidates
            else 0.0
        )

        cost = (
            new_count / new_true
            if new_true
            else float("inf")
        )

        print()
        print("=" * 90)
        print("COMBINED RECALL RESULT")
        print("=" * 90)

        print(
            f"Base candidates        : {base_count:,}"
        )

        print(
            f"Base true pairs        : {base_true:,}"
        )

        print(
            f"New candidates         : {new_count:,}"
        )

        print(
            f"New true pairs         : {new_true:,}"
        )

        print(
            f"Candidates / new true : {cost:.2f}"
        )

        print(
            f"Incremental recall     : "
            f"{incremental_recall:.6%}"
        )

        print(
            f"Incremental purity     : "
            f"{incremental_purity:.6%}"
        )

        print(
            f"FINAL candidate count  : "
            f"{combined_candidates:,}"
        )

        print(
            f"FINAL true pairs       : "
            f"{combined_true:,}"
        )

        print(
            f"PROJECTED recall       : "
            f"{combined_recall:.6%}"
        )

        print(
            f"PROJECTED purity       : "
            f"{combined_purity:.6%}"
        )

        print("=" * 90)

    finally:
        con.close()


if __name__ == "__main__":
    main()
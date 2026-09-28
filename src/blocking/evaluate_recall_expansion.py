from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]

BLOCKING = ROOT / "artifacts" / "blocking"

UNION_DIR = BLOCKING / "union"

EXPERIMENTS = BLOCKING / "experiments"

TMP = BLOCKING / "duckdb_tmp"

GT = BLOCKING / "ground_truth_pairs.parquet"

BASE = UNION_DIR / "train_optimized_candidates.parquet"

REPORT = EXPERIMENTS / "recall_expansion_report.csv"


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def configure(con: duckdb.DuckDBPyConnection) -> None:

    con.execute("SET threads=8")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")

    TMP.mkdir(
        parents=True,
        exist_ok=True,
    )

    con.execute(
        f"SET temp_directory='{sql_path(TMP)}'"
    )

    con.execute(
        "SET enable_progress_bar=true"
    )


def scalar(
    con: duckdb.DuckDBPyConnection,
    sql: str,
):
    return con.execute(sql).fetchone()[0]


def evaluate_one(
    con: duckdb.DuckDBPyConnection,
    base: Path,
    experiment: Path,
) -> dict:

    print("\n" + "-" * 80)
    print(f"EVALUATING: {experiment.name}")
    print("-" * 80)

    b = sql_path(base)
    e = sql_path(experiment)
    g = sql_path(GT)

    experiment_count = scalar(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{e}')
        """,
    )

    base_count = scalar(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{b}')
        """,
    )

    total_true = scalar(
        con,
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{g}')
        """,
    )

    print("[1/4] Evaluating base true pairs...")

    base_true = scalar(
        con,
        f"""
        SELECT COUNT(*)

        FROM read_parquet('{b}') c

        INNER JOIN read_parquet('{g}') g

          ON c.source1_entity_id = g.source1_entity_id
         AND c.matched_entity_id = g.matched_entity_id
         AND c.matched_source = g.matched_source
        """,
    )


    print("[2/4] Finding NEW candidates...")

    con.execute(
        "DROP TABLE IF EXISTS incremental_candidates"
    )

    con.execute(
        f"""
        CREATE TEMP TABLE incremental_candidates AS

        SELECT DISTINCT
            e.source1_entity_id,
            e.matched_entity_id,
            e.matched_source

        FROM read_parquet('{e}') e

        ANTI JOIN read_parquet('{b}') base

          ON e.source1_entity_id = base.source1_entity_id
         AND e.matched_entity_id = base.matched_entity_id
         AND e.matched_source = base.matched_source
        """
    )

    new_count = scalar(
        con,
        """
        SELECT COUNT(*)
        FROM incremental_candidates
        """,
    )

    
    print("[3/4] Measuring NEW true pairs...")

    new_true = scalar(
        con,
        f"""
        SELECT COUNT(*)

        FROM incremental_candidates c

        INNER JOIN read_parquet('{g}') gt

          ON c.source1_entity_id = gt.source1_entity_id
         AND c.matched_entity_id = gt.matched_entity_id
         AND c.matched_source = gt.matched_source
        """,
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

    candidates_per_new_true = (
        new_count / new_true
        if new_true
        else float("inf")
    )

    row = {
        "method": experiment.stem,
        "base_candidates": base_count,
        "base_true": base_true,
        "experiment_candidates": experiment_count,
        "new_candidates": new_count,
        "new_true_pairs": new_true,
        "incremental_recall": incremental_recall,
        "incremental_purity": incremental_purity,
        "candidates_per_new_true": candidates_per_new_true,
        "projected_candidates": combined_candidates,
        "projected_true_pairs": combined_true,
        "projected_recall": combined_recall,
        "projected_purity": combined_purity,
    }

    print("[4/4] Results")
    print()

    print(
        f"{'Base candidates':30s}: "
        f"{base_count:,}"
    )

    print(
        f"{'Base true pairs':30s}: "
        f"{base_true:,}"
    )

    print(
        f"{'Experiment candidates':30s}: "
        f"{experiment_count:,}"
    )

    print(
        f"{'NEW candidates':30s}: "
        f"{new_count:,}"
    )

    print(
        f"{'NEW true pairs':30s}: "
        f"{new_true:,}"
    )

    print(
        f"{'Incremental recall':30s}: "
        f"{incremental_recall:.6%}"
    )

    print(
        f"{'Incremental purity':30s}: "
        f"{incremental_purity:.6%}"
    )

    print(
        f"{'Candidates / NEW true':30s}: "
        f"{candidates_per_new_true:.2f}"
    )

    print(
        f"{'Projected recall':30s}: "
        f"{combined_recall:.6%}"
    )

    print(
        f"{'Projected purity':30s}: "
        f"{combined_purity:.6%}"
    )

    return row


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--base",
        default=str(BASE),
        help="Frozen optimized base candidate parquet",
    )

    parser.add_argument(
        "--experiments",
        default="ascii,numeric,rare_name",
        help="Comma-separated experiment names",
    )

    args = parser.parse_args()

    base = Path(args.base)

    if not base.is_absolute():
        base = ROOT / base

    if not base.exists():
        raise FileNotFoundError(
            f"Base candidate file not found:\n{base}"
        )

    if not GT.exists():
        raise FileNotFoundError(
            f"Ground truth file not found:\n{GT}"
        )

    names = [
        x.strip().lower()
        for x in args.experiments.split(",")
        if x.strip()
    ]

    experiment_files = []

    patterns = {
        "ascii": "train_ascii_exact_candidates.parquet",
        "numeric": "train_numeric_anchor_candidates.parquet",
        "rare_name": "train_rare_name_strict_candidates.parquet",
    }

    for name in names:

        if name not in patterns:
            raise ValueError(
                f"Unknown experiment: {name}"
            )

        path = EXPERIMENTS / patterns[name]

        if not path.exists():
            raise FileNotFoundError(
                f"""
Experiment file not found:

{name}
{path}

Run the corresponding expansion experiment first.
"""
            )

        experiment_files.append(path)

    con = duckdb.connect()

    try:

        configure(con)

        rows = []

        for experiment in experiment_files:

            rows.append(
                evaluate_one(
                    con,
                    base,
                    experiment,
                )
            )


        REPORT.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        con.execute(
            "DROP TABLE IF EXISTS recall_report"
        )

        con.execute(
            """
            CREATE TEMP TABLE recall_report
            (
                method VARCHAR,

                base_candidates BIGINT,
                base_true BIGINT,

                experiment_candidates BIGINT,

                new_candidates BIGINT,
                new_true_pairs BIGINT,

                incremental_recall DOUBLE,
                incremental_purity DOUBLE,

                candidates_per_new_true DOUBLE,

                projected_candidates BIGINT,
                projected_true_pairs BIGINT,

                projected_recall DOUBLE,
                projected_purity DOUBLE
            )
            """
        )

        for row in rows:

            con.execute(
                """
                INSERT INTO recall_report
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    row["method"],
                    row["base_candidates"],
                    row["base_true"],
                    row["experiment_candidates"],
                    row["new_candidates"],
                    row["new_true_pairs"],
                    row["incremental_recall"],
                    row["incremental_purity"],
                    row["candidates_per_new_true"],
                    row["projected_candidates"],
                    row["projected_true_pairs"],
                    row["projected_recall"],
                    row["projected_purity"],
                ],
            )

        con.execute(
            f"""
            COPY recall_report

            TO '{sql_path(REPORT)}'

            (FORMAT CSV, HEADER)
            """
        )

    
        print("\n" + "=" * 100)
        print("RECALL EXPANSION SUMMARY")
        print("=" * 100)

        print(
            f"{'METHOD':22s}"
            f"{'NEW CAND':>15s}"
            f"{'NEW TRUE':>15s}"
            f"{'INC REC':>12s}"
            f"{'INC PUR':>12s}"
            f"{'C/TRUE':>12s}"
            f"{'PROJ REC':>12s}"
        )

        summary = con.execute(
            """
            SELECT
                method,
                new_candidates,
                new_true_pairs,
                incremental_recall,
                incremental_purity,
                candidates_per_new_true,
                projected_recall

            FROM recall_report

            ORDER BY
                incremental_purity DESC
            """
        ).fetchall()

        for row in summary:

            print(
                f"{row[0]:22s}"
                f"{row[1]:15,d}"
                f"{row[2]:15,d}"
                f"{row[3]:12.4%}"
                f"{row[4]:12.4%}"
                f"{row[5]:12.1f}"
                f"{row[6]:12.4%}"
            )

        print()
        print(f"Saved report: {REPORT}")

    finally:
        con.close()


if __name__ == "__main__":
    main()

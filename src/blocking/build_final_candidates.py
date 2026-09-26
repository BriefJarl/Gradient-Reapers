from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


# ============================================================
# AMAZON ML CHALLENGE 2026
# FINAL CANDIDATE BUILDER
#
# Takes:
#
#   frozen 31.22M base
#          +
#   selected recall-expansion experiments
#
# and produces:
#
#   train_final_candidates.parquet
#
# The base candidate pool is NEVER modified.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

BLOCKING = ROOT / "artifacts" / "blocking"

UNION_DIR = BLOCKING / "union"

EXPERIMENTS = BLOCKING / "experiments"

TMP = BLOCKING / "duckdb_tmp"

TRAIN_BASE = (
    UNION_DIR /
    "train_optimized_candidates.parquet"
)

TRAIN_OUT = (
    UNION_DIR /
    "train_final_candidates.parquet"
)

GT = (
    BLOCKING /
    "ground_truth_pairs.parquet"
)


EXPERIMENT_FILES = {
    "ascii":
        "train_ascii_exact_candidates.parquet",

    "numeric":
        "train_numeric_anchor_candidates.parquet",

    "rare_name":
        "train_rare_name_strict_candidates.parquet",
}


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace(
        "\\",
        "/",
    ).replace(
        "'",
        "''",
    )


def configure(con: duckdb.DuckDBPyConnection) -> None:

    con.execute("SET threads=8")

    con.execute(
        "SET memory_limit='8GB'"
    )

    con.execute(
        "SET preserve_insertion_order=false"
    )

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


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--methods",
        required=True,
        help=(
            "Comma-separated methods: "
            "ascii,numeric,rare_name"
        ),
    )

    parser.add_argument(
        "--base",
        default=str(TRAIN_BASE),
    )

    parser.add_argument(
        "--output",
        default=str(TRAIN_OUT),
    )

    args = parser.parse_args()

    base = Path(args.base)

    if not base.is_absolute():
        base = ROOT / base

    output = Path(args.output)

    if not output.is_absolute():
        output = ROOT / output

    if not base.exists():
        raise FileNotFoundError(
            f"Base candidate file not found:\n{base}"
        )

    methods = [
        x.strip().lower()
        for x in args.methods.split(",")
        if x.strip()
    ]

    if not methods:
        raise ValueError(
            "At least one method is required."
        )

    unknown = (
        set(methods)
        -
        set(EXPERIMENT_FILES)
    )

    if unknown:
        raise ValueError(
            f"Unknown methods: {sorted(unknown)}"
        )

    experiment_paths = []

    for method in methods:

        path = (
            EXPERIMENTS /
            EXPERIMENT_FILES[method]
        )

        if not path.exists():
            raise FileNotFoundError(
                f"""
Experiment file missing:

Method:
{method}

File:
{path}

Run expand_recall.py first.
"""
            )

        experiment_paths.append(
            (method, path)
        )

    con = duckdb.connect()

    try:

        configure(con)

        print("=" * 90)
        print("BUILDING FINAL TRAIN CANDIDATE SET")
        print("=" * 90)

        print(f"Base:")
        print(base)

        print()
        print(
            "Selected methods: "
            +
            ", ".join(methods)
        )

        print()
        print(f"Output:")
        print(output)

        # ----------------------------------------------------
        # Base relation
        # ----------------------------------------------------

        base_sql = sql_path(base)

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW base_candidates AS

            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                blocking_mask,
                num_blocking_methods,
                blocking_methods

            FROM read_parquet(
                '{base_sql}'
            )
            """
        )

        base_count = con.execute(
            """
            SELECT COUNT(*)
            FROM base_candidates
            """
        ).fetchone()[0]

        print()
        print(
            f"Base candidates: {base_count:,}"
        )

        # ----------------------------------------------------
        # Build one UNION ALL relation for the selected
        # experiment candidates.
        #
        # We aggregate provenance before removing base
        # duplicates.
        # ----------------------------------------------------

        experiment_selects = []

        for method, path in experiment_paths:

            p = sql_path(path)

            experiment_selects.append(
                f"""
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    blocking_methods

                FROM read_parquet(
                    '{p}'
                )
                """
            )

        extra_union_sql = "\nUNION ALL\n".join(
            experiment_selects
        )

        print()
        print(
            "Preparing selected experiment candidates..."
        )

        con.execute(
            f"""
            CREATE OR REPLACE TEMP VIEW selected_extra_raw AS

            {extra_union_sql}
            """
        )

        # ----------------------------------------------------
        # Deduplicate experiment candidates.
        #
        # A pair may be discovered by both ASCII and numeric.
        # Preserve combined blocking provenance.
        # ----------------------------------------------------

        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW selected_extra AS

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

            FROM selected_extra_raw

            GROUP BY
                source1_entity_id,
                matched_entity_id,
                matched_source
            """
        )

        # ----------------------------------------------------
        # Remove everything already present in base.
        #
        # This is the key operation.
        # ----------------------------------------------------

        print()
        print(
            "Removing candidates already present "
            "in the frozen base..."
        )

        con.execute(
            """
            CREATE OR REPLACE TEMP VIEW new_candidates AS

            SELECT
                e.source1_entity_id,
                e.matched_entity_id,
                e.matched_source,
                e.blocking_mask,
                e.num_blocking_methods,
                e.blocking_methods

            FROM selected_extra e

            ANTI JOIN base_candidates b

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
            FROM new_candidates
            """
        ).fetchone()[0]

        print(
            f"New candidates: {new_count:,}"
        )

        total_count = (
            base_count +
            new_count
        )

        # ----------------------------------------------------
        # Write final Parquet.
        # ----------------------------------------------------

        if output.exists():
            output.unlink()

        output.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        print()
        print(
            "Writing final candidate Parquet..."
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

                FROM base_candidates

                UNION ALL

                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    num_blocking_methods,
                    blocking_methods

                FROM new_candidates
            )

            TO '{sql_path(output)}'

            (
                FORMAT PARQUET,
                COMPRESSION ZSTD
            )
            """
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        print()
        print("-" * 90)
        print("FINAL CANDIDATE VALIDATION")
        print("-" * 90)

        final_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet(
                '{sql_path(output)}'
            )
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

                FROM read_parquet(
                    '{sql_path(output)}'
                )

                GROUP BY
                    source1_entity_id,
                    matched_entity_id,
                    matched_source

                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]

        print(
            f"Base candidates        : {base_count:,}"
        )

        print(
            f"New candidates         : {new_count:,}"
        )

        print(
            f"Final candidates       : {final_count:,}"
        )

        print(
            f"Expected final count   : {total_count:,}"
        )

        print(
            f"Duplicate pair groups  : {duplicate_count:,}"
        )

        if final_count != total_count:
            raise RuntimeError(
                "FINAL ROW COUNT MISMATCH"
            )

        if duplicate_count != 0:
            raise RuntimeError(
                "DUPLICATE CANDIDATE PAIRS DETECTED"
            )

        # ----------------------------------------------------
        # Ground-truth evaluation
        # ----------------------------------------------------

        if GT.exists():

            gt_sql = sql_path(GT)
            out_sql = sql_path(output)

            true_pairs = con.execute(
                f"""
                SELECT COUNT(*)

                FROM read_parquet(
                    '{out_sql}'
                ) c

                INNER JOIN read_parquet(
                    '{gt_sql}'
                ) g

                  ON c.source1_entity_id =
                     g.source1_entity_id

                 AND c.matched_entity_id =
                     g.matched_entity_id

                 AND c.matched_source =
                     g.matched_source
                """
            ).fetchone()[0]

            total_true = con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet(
                    '{gt_sql}'
                )
                """
            ).fetchone()[0]

            recall = (
                true_pairs / total_true
                if total_true
                else 0.0
            )

            purity = (
                true_pairs / final_count
                if final_count
                else 0.0
            )

            print()
            print(
                f"True pairs recovered   : "
                f"{true_pairs:,}"
            )

            print(
                f"Total true pairs       : "
                f"{total_true:,}"
            )

            print(
                f"Candidate recall       : "
                f"{recall:.6%}"
            )

            print(
                f"Candidate purity       : "
                f"{purity:.6%}"
            )

        print()
        print("=" * 90)
        print("FINAL CANDIDATE BUILD COMPLETE")
        print("=" * 90)
        print()
        print(
            f"Saved:\n{output}"
        )

    finally:
        con.close()


if __name__ == "__main__":
    main()
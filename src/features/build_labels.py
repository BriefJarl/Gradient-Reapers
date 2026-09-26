from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

FEATURE_FILE = (
    ROOT
    / "artifacts"
    / "features"
    / "train_features.parquet"
)

GROUND_TRUTH = (
    ROOT
    / "artifacts"
    / "blocking"
    / "ground_truth_pairs.parquet"
)

OUTPUT = (
    ROOT
    / "artifacts"
    / "features"
    / "train_features_labeled.parquet"
)

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def main() -> None:

    print("=" * 80)
    print("BUILDING TRAINING LABELS")
    print("=" * 80)

    for path in (
        FEATURE_FILE,
        GROUND_TRUTH,
    ):
        if not path.exists():
            raise FileNotFoundError(
                f"Required file not found:\n{path}"
            )

    if OUTPUT.exists():
        OUTPUT.unlink()

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

        temp_dir = (
            ROOT
            / "artifacts"
            / "blocking"
            / "duckdb_tmp"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        con.execute(
            f"SET temp_directory = "
            f"'{sql_quote(temp_dir)}'"
        )

        feature_sql = sql_quote(FEATURE_FILE)
        gt_sql = sql_quote(GROUND_TRUTH)
        output_sql = sql_quote(OUTPUT)

        # ----------------------------------------------------
        # Create labels
        # ----------------------------------------------------

        print()
        print("Joining ground truth to candidate features...")

        con.execute(
            f"""
            COPY
            (
                SELECT
                    f.*,

                    CASE
                        WHEN g.source1_entity_id IS NOT NULL
                        THEN CAST(1 AS TINYINT)
                        ELSE CAST(0 AS TINYINT)
                    END AS label

                FROM read_parquet('{feature_sql}') f

                LEFT JOIN
                (
                    SELECT DISTINCT
                        source1_entity_id,
                        matched_entity_id,
                        matched_source
                    FROM read_parquet('{gt_sql}')
                ) g

                    ON f.source1_entity_id
                        = g.source1_entity_id

                   AND f.matched_entity_id
                        = g.matched_entity_id

                   AND f.matched_source
                        = g.matched_source
            )
            TO '{output_sql}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        total = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output_sql}')
            """
        ).fetchone()[0]

        positives = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{output_sql}')
            WHERE label = 1
            """
        ).fetchone()[0]

        negatives = total - positives

        print()
        print("=" * 80)
        print("LABEL SUMMARY")
        print("=" * 80)

        print(f"Total candidate pairs : {total:,}")
        print(f"Positive pairs        : {positives:,}")
        print(f"Negative pairs        : {negatives:,}")

        print(
            f"Positive rate         : "
            f"{positives / total:.6%}"
        )

        # ----------------------------------------------------
        # Source-wise labels
        # ----------------------------------------------------

        print()
        print("SOURCE-WISE LABELS")

        rows = con.execute(
            f"""
            SELECT
                matched_source,
                COUNT(*) AS candidates,

                SUM(
                    CASE
                        WHEN label = 1
                        THEN 1 ELSE 0
                    END
                ) AS positives

            FROM read_parquet('{output_sql}')
            GROUP BY matched_source
            ORDER BY matched_source
            """
        ).fetchall()

        for source, candidates, pos in rows:

            print(
                f"{source}: "
                f"{candidates:,} candidates, "
                f"{pos:,} positives, "
                f"{pos / candidates:.6%} positive rate"
            )

        print()
        print(f"OUTPUT:")
        print(OUTPUT)

        print()
        print("=" * 80)
        print("TRAINING LABELS COMPLETE")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
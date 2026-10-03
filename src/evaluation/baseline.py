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
    / "evaluation"
    / "baseline_predictions.parquet"
)

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def main() -> None:

    print("=" * 80)
    print("DETERMINISTIC BASELINE")
    print("=" * 80)

    if not FEATURE_FILE.exists():
        raise FileNotFoundError(
            f"Feature file not found:\n{FEATURE_FILE}"
        )

    if not GROUND_TRUTH.exists():
        raise FileNotFoundError(
            f"Ground truth not found:\n{GROUND_TRUTH}"
        )

    OUTPUT.parent.mkdir(
        parents=True,
        exist_ok=True,
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

        feature_sql = sql_quote(FEATURE_FILE)
        gt_sql = sql_quote(GROUND_TRUTH)
        output_sql = sql_quote(OUTPUT)

        print()
        print("Rule:")
        print(
            "country_exact = 1 AND "
            "(name_exact OR name_compact_exact OR "
            "address_exact OR address_compact_exact)"
        )

        con.execute(
            f"""
            COPY
            (
                SELECT
                    f.source1_entity_id,
                    f.matched_entity_id,
                    f.matched_source,

                    CASE
                        WHEN
                            f.country_exact = 1
                            AND
                            (
                                f.name_exact = 1
                                OR f.name_compact_exact = 1
                                OR f.address_exact = 1
                                OR f.address_compact_exact = 1
                            )
                        THEN 1
                        ELSE 0
                    END AS prediction,

                    CASE
                        WHEN g.source1_entity_id IS NOT NULL
                        THEN 1
                        ELSE 0
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

        stats = con.execute(
            f"""
            SELECT
                SUM(
                    CASE
                        WHEN prediction = 1
                         AND label = 1
                        THEN 1 ELSE 0
                    END
                ) AS tp,

                SUM(
                    CASE
                        WHEN prediction = 1
                         AND label = 0
                        THEN 1 ELSE 0
                    END
                ) AS fp,

                SUM(
                    CASE
                        WHEN prediction = 0
                         AND label = 1
                        THEN 1 ELSE 0
                    END
                ) AS fn

            FROM read_parquet('{output_sql}')
            """
        ).fetchone()

        tp = stats[0] or 0
        fp = stats[1] or 0
        fn = stats[2] or 0

        precision = (
            tp / (tp + fp)
            if tp + fp > 0
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn > 0
            else 0.0
        )

        beta = 0.5

        f05 = (
            (1 + beta**2)
            * precision
            * recall
            /
            (
                beta**2 * precision
                + recall
            )
            if precision > 0 and recall > 0
            else 0.0
        )

        print()
        print("=" * 80)
        print("BASELINE DIAGNOSTIC RESULTS")
        print("=" * 80)

        print()
        print(f"TP        : {tp:,}")
        print(f"FP        : {fp:,}")
        print(f"FN        : {fn:,}")
        print(f"Precision : {precision:.6f}")
        print(f"Recall    : {recall:.6f}")
        print(f"F0.5      : {f05:.6f}")

        print()
        print(
            "NOTE: These are pair-level diagnostic metrics. "
            "Use the competition's official evaluation logic "
            "for the final Macro-F0.5 comparison."
        )

        print()
        print(f"Output: {OUTPUT}")

    finally:
        con.close()


if __name__ == "__main__":
    main()

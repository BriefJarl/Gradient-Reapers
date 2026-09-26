from __future__ import annotations

"""
Entity-level Macro F0.5 evaluation and threshold search for Amazon ML Challenge 2026.

Input:
    artifacts/features/final_train/valid_predictions.parquet
    artifacts/blocking/ground_truth_pairs.parquet
    artifacts/features/final_train/valid_split.parquet

Output:
    artifacts/models/entity_threshold_results.parquet
    artifacts/models/entity_threshold.json

Important:
- The metric is computed per Source-1 entity.
- Validation entities come from valid_split, so entities with zero predicted
  matches at a threshold are retained in the macro average.
- Pair identity is (source1_entity_id, matched_entity_id, matched_source).
- No pandas; DuckDB performs the large aggregation.
"""

import argparse
import json
import math
import os
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "artifacts" / "features" / "final_train"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
MODEL_DIR = ROOT / "artifacts" / "models"
TMP_DIR = BLOCKING_DIR / "duckdb_tmp"

PREDICTIONS = FEATURE_DIR / "valid_predictions.parquet"
VALID_SPLIT = FEATURE_DIR / "valid_split.parquet"
GROUND_TRUTH = BLOCKING_DIR / "ground_truth_pairs.parquet"

RESULTS = MODEL_DIR / "entity_threshold_results.parquet"
BEST_JSON = MODEL_DIR / "entity_threshold.json"

MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY", "8GB")
THREADS = int(
    os.environ.get(
        "DUCKDB_THREADS",
        str(max(4, min(12, (os.cpu_count() or 10) - 2))),
    )
)

BETA2 = 0.25


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def configure(con: duckdb.DuckDBPyConnection) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{sql_quote(TMP_DIR)}'")


def parse_thresholds(spec: str) -> list[float]:
    values = []
    for item in spec.split(","):
        value = float(item.strip())
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"Threshold must be in [0,1]: {value}")
        values.append(value)

    values = sorted(set(round(v, 6) for v in values))
    if not values:
        raise ValueError("No thresholds supplied.")
    return values


def f05(tp: float, fp: float, fn: float) -> float:
    precision_den = tp + fp
    recall_den = tp + fn

    if precision_den == 0 and recall_den == 0:
        return 1.0

    precision = tp / precision_den if precision_den else 0.0
    recall = tp / recall_den if recall_den else 0.0

    denom = BETA2 * precision + recall
    if denom == 0:
        return 0.0

    return (1.0 + BETA2) * precision * recall / denom


def evaluate_thresholds(
    con: duckdb.DuckDBPyConnection,
    thresholds: list[float],
) -> list[tuple]:
    # Materialize the threshold list as a tiny DuckDB relation.
    values_sql = ",".join(f"({v:.6f})" for v in thresholds)

    con.execute("DROP VIEW IF EXISTS valid_s1")
    con.execute("DROP VIEW IF EXISTS gt_valid")
    con.execute("DROP VIEW IF EXISTS pred")

    con.execute(
        f"""
        CREATE TEMP VIEW valid_s1 AS
        SELECT DISTINCT source1_entity_id
        FROM read_parquet('{sql_quote(VALID_SPLIT)}')
        """
    )

    con.execute(
        f"""
        CREATE TEMP VIEW gt_valid AS
        SELECT
            g.source1_entity_id,
            g.matched_entity_id,
            g.matched_source
        FROM read_parquet('{sql_quote(GROUND_TRUTH)}') g
        INNER JOIN valid_s1 v
          ON g.source1_entity_id = v.source1_entity_id
        """
    )

    con.execute(
        f"""
        CREATE TEMP VIEW pred AS
        SELECT
            p.source1_entity_id,
            p.matched_entity_id,
            p.matched_source,
            CAST(p.score AS DOUBLE) AS score,
            CASE WHEN g.source1_entity_id IS NOT NULL THEN 1 ELSE 0 END AS is_tp
        FROM read_parquet('{sql_quote(PREDICTIONS)}') p
        LEFT JOIN gt_valid g
          ON p.source1_entity_id = g.source1_entity_id
         AND p.matched_entity_id = g.matched_entity_id
         AND p.matched_source = g.matched_source
        """
    )

    # For each threshold:
    #   predicted_count = number of selected candidate pairs
    #   tp_count        = selected true pairs
    # Then FP = predicted - TP and FN = true_count - TP.
    #
    # Cross joining 8.4M predictions with 19 coarse thresholds is intentional
    # and stays inside DuckDB; no Python row loop is used.
    con.execute("DROP TABLE IF EXISTS threshold_metrics")

    threshold_values = f"""
        SELECT * FROM (VALUES {values_sql}) AS t(threshold)
    """

    print()
    print(f"Evaluating {len(thresholds)} thresholds...")
    print("This is the main DuckDB aggregation pass.")

    con.execute(
        f"""
        CREATE TEMP TABLE threshold_metrics AS
        WITH
        thresholds AS (
            {threshold_values}
        ),
        true_counts AS (
            SELECT
                source1_entity_id,
                COUNT(*)::BIGINT AS true_count
            FROM gt_valid
            GROUP BY source1_entity_id
        ),
        predicted AS (
            SELECT
                p.source1_entity_id,
                t.threshold,
                COUNT(*)::BIGINT AS predicted_count,
                SUM(p.is_tp)::BIGINT AS tp
            FROM pred p
            CROSS JOIN thresholds t
            WHERE p.score >= t.threshold
            GROUP BY p.source1_entity_id, t.threshold
        ),
        all_entities AS (
            SELECT
                v.source1_entity_id,
                t.threshold,
                COALESCE(tc.true_count, 0)::BIGINT AS true_count
            FROM valid_s1 v
            CROSS JOIN thresholds t
            LEFT JOIN true_counts tc
              ON v.source1_entity_id = tc.source1_entity_id
        ),
        entity_scores AS (
            SELECT
                a.threshold,
                a.source1_entity_id,
                a.true_count,
                COALESCE(p.predicted_count, 0)::BIGINT AS predicted_count,
                COALESCE(p.tp, 0)::BIGINT AS tp
            FROM all_entities a
            LEFT JOIN predicted p
              ON a.source1_entity_id = p.source1_entity_id
             AND a.threshold = p.threshold
        ),
        scored AS (
            SELECT
                threshold,
                source1_entity_id,
                predicted_count,
                true_count,
                tp,
                predicted_count - tp AS fp,
                true_count - tp AS fn,
                CASE
                    WHEN predicted_count = 0 AND true_count = 0 THEN 1.0
                    WHEN
                        0.25 * (
                            CASE
                                WHEN predicted_count > 0
                                THEN tp::DOUBLE / predicted_count
                                ELSE 0.0
                            END
                        )
                        +
                        (
                            CASE
                                WHEN true_count > 0
                                THEN tp::DOUBLE / true_count
                                ELSE 0.0
                            END
                        )
                        = 0
                    THEN 0.0
                    ELSE
                        1.25
                        *
                        (
                            CASE
                                WHEN predicted_count > 0
                                THEN tp::DOUBLE / predicted_count
                                ELSE 0.0
                            END
                        )
                        *
                        (
                            CASE
                                WHEN true_count > 0
                                THEN tp::DOUBLE / true_count
                                ELSE 0.0
                            END
                        )
                        /
                        (
                            0.25 * (
                                CASE
                                    WHEN predicted_count > 0
                                    THEN tp::DOUBLE / predicted_count
                                    ELSE 0.0
                                END
                            )
                            +
                            (
                                CASE
                                    WHEN true_count > 0
                                    THEN tp::DOUBLE / true_count
                                    ELSE 0.0
                                END
                            )
                        )
                END AS f05
            FROM entity_scores
        )
        SELECT
            threshold,
            COUNT(*)::BIGINT AS validation_s1_count,
            SUM(tp)::BIGINT AS tp,
            SUM(fp)::BIGINT AS fp,
            SUM(fn)::BIGINT AS fn,
            SUM(predicted_count)::BIGINT AS predicted_pairs,
            SUM(true_count)::BIGINT AS true_pairs,
            AVG(f05)::DOUBLE AS macro_f05,
            AVG(
                CASE
                    WHEN predicted_count > 0
                    THEN tp::DOUBLE / predicted_count
                    ELSE 0.0
                END
            ) AS macro_precision,
            AVG(
                CASE
                    WHEN true_count > 0
                    THEN tp::DOUBLE / true_count
                    ELSE 0.0
                END
            ) AS macro_recall
        FROM scored
        GROUP BY threshold
        ORDER BY threshold
        """
    )

    rows = con.execute(
        """
        SELECT
            threshold,
            validation_s1_count,
            tp,
            fp,
            fn,
            predicted_pairs,
            true_pairs,
            macro_f05,
            macro_precision,
            macro_recall
        FROM threshold_metrics
        ORDER BY threshold
        """
    ).fetchall()

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--thresholds",
        default="0.05,0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90,0.95",
        help="Comma-separated threshold grid.",
    )
    parser.add_argument(
        "--refine",
        action="store_true",
        help="After coarse search, evaluate a fine grid around the best threshold.",
    )
    args = parser.parse_args()

    for path in (PREDICTIONS, VALID_SPLIT, GROUND_TRUTH):
        if not path.exists():
            raise FileNotFoundError(path)

    thresholds = parse_thresholds(args.thresholds)

    print("=" * 88)
    print("AMAZON ML CHALLENGE 2026")
    print("ENTITY-LEVEL MACRO F0.5 THRESHOLD EVALUATOR")
    print("=" * 88)
    print(f"Predictions : {PREDICTIONS}")
    print(f"Validation  : {VALID_SPLIT}")
    print(f"Ground truth: {GROUND_TRUTH}")
    print(f"DuckDB      : {THREADS} threads / {MEMORY_LIMIT}")
    print(f"Beta        : 0.5")
    print(f"Thresholds  : {len(thresholds)}")

    con = duckdb.connect()

    try:
        configure(con)

        rows = evaluate_thresholds(con, thresholds)

        if args.refine:
            best = max(rows, key=lambda r: r[7])
            best_threshold = float(best[0])

            low = max(0.0, best_threshold - 0.05)
            high = min(1.0, best_threshold + 0.05)

            fine = [
                round(low + i * 0.005, 6)
                for i in range(int(round((high - low) / 0.005)) + 1)
            ]

            fine_rows = evaluate_thresholds(con, sorted(set(fine)))

            # Keep the coarse results plus all fine results.
            by_threshold = {round(float(r[0]), 6): r for r in rows}
            for r in fine_rows:
                by_threshold[round(float(r[0]), 6)] = r

            rows = [by_threshold[k] for k in sorted(by_threshold)]

        # Write results using parameterized INSERTs rather than constructing
        # a giant VALUES (...) SQL string. This avoids Decimal/repr parsing
        # issues and is safer for future threshold grids.
        con.execute("DROP TABLE IF EXISTS final_threshold_results")
        con.execute(
            """
            CREATE TEMP TABLE final_threshold_results (
                threshold DOUBLE,
                validation_s1_count BIGINT,
                tp BIGINT,
                fp BIGINT,
                fn BIGINT,
                predicted_pairs BIGINT,
                true_pairs BIGINT,
                macro_f05 DOUBLE,
                macro_precision DOUBLE,
                macro_recall DOUBLE
            )
            """
        )

        con.executemany(
            """
            INSERT INTO final_threshold_results
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                (
                    float(row[0]),
                    int(row[1]),
                    int(row[2]),
                    int(row[3]),
                    int(row[4]),
                    int(row[5]),
                    int(row[6]),
                    float(row[7]),
                    float(row[8]),
                    float(row[9]),
                )
                for row in rows
            ],
        )

        if RESULTS.exists():
            RESULTS.unlink()

        con.execute(
            f"""
            COPY final_threshold_results
            TO '{sql_quote(RESULTS)}'
            (FORMAT PARQUET, COMPRESSION ZSTD)
            """
        )

        best = max(rows, key=lambda r: r[7])

        result = {
            "metric": "macro_f0.5",
            "beta": 0.5,
            "best_threshold": float(best[0]),
            "macro_f05": float(best[7]),
            "macro_precision": float(best[8]),
            "macro_recall": float(best[9]),
            "validation_s1_count": int(best[1]),
            "tp": int(best[2]),
            "fp": int(best[3]),
            "fn": int(best[4]),
            "predicted_pairs": int(best[5]),
            "true_pairs": int(best[6]),
            "threshold_count": len(rows),
        }

        BEST_JSON.write_text(
            json.dumps(result, indent=2),
            encoding="utf-8",
        )

        print()
        print("THRESHOLD RESULTS")
        print("-" * 88)
        for row in rows:
            print(
                f"threshold={row[0]:.3f}  "
                f"F0.5={row[7]:.6f}  "
                f"macroP={row[8]:.6f}  "
                f"macroR={row[9]:.6f}  "
                f"pred={row[5]:,}"
            )

        print()
        print("=" * 88)
        print("BEST ENTITY-LEVEL THRESHOLD")
        print("=" * 88)
        print(f"Threshold       : {result['best_threshold']:.6f}")
        print(f"Macro F0.5      : {result['macro_f05']:.6f}")
        print(f"Macro Precision : {result['macro_precision']:.6f}")
        print(f"Macro Recall    : {result['macro_recall']:.6f}")
        print(f"TP              : {result['tp']:,}")
        print(f"FP              : {result['fp']:,}")
        print(f"FN              : {result['fn']:,}")
        print(f"Predicted pairs : {result['predicted_pairs']:,}")
        print(f"Validation S1   : {result['validation_s1_count']:,}")
        print()
        print(f"Results : {RESULTS}")
        print(f"Best    : {BEST_JSON}")
        print("=" * 88)

    finally:
        con.close()


if __name__ == "__main__":
    main()

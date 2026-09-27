from __future__ import annotations

"""
Ensemble Weight & Threshold Tuning Experiment
Evaluates individual models (LGB, CB, XGB) and candidate probability blends
on scored_valid_pairs_phase3_fixed.parquet against the exact entity-level Macro F0.5
competition metric.

PROTECTED BASELINE REFERENCE:
Macro F0.5 ≈ 0.7659 at threshold ≈ 0.585 (trained on final_lightgbm.txt).
"""

import argparse
from pathlib import Path
import sys
import time
import duckdb

ROOT = Path(__file__).resolve().parents[2]
PRED_PATH = ROOT / "artifacts" / "features" / "scored_valid_pairs_phase3_fixed.parquet"
VALID_PATH = ROOT / "artifacts" / "features" / "final_train" / "valid_split.parquet"
GT_PATH = ROOT / "artifacts" / "blocking" / "ground_truth_pairs.parquet"
OUT_RESULTS = ROOT / "artifacts" / "models" / "ensemble_weight_results.parquet"

PROTECTED_BASELINE_F05 = 0.7659

COARSE_THRESHOLDS = [
    0.35,
    0.40,
    0.45,
    0.50,
    0.525,
    0.55,
    0.575,
    0.60,
    0.625,
    0.65,
    0.675,
    0.70,
    0.75,
    0.80,
]

REQUIRED_COLUMNS = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "label",
    "p_lgb",
    "p_cb",
    "p_xgb",
    "pred_prob",
}

MODELS_AND_BLENDS = [
    {
        "model": "LGB only",
        "weights": "(1.00, 0.00, 0.00)",
        "score_expr": "p.p_lgb",
        "w_lgb": 1.00,
        "w_cb": 0.00,
        "w_xgb": 0.00,
    },
    {
        "model": "CatBoost only",
        "weights": "(0.00, 1.00, 0.00)",
        "score_expr": "p.p_cb",
        "w_lgb": 0.00,
        "w_cb": 1.00,
        "w_xgb": 0.00,
    },
    {
        "model": "XGBoost only",
        "weights": "(0.00, 0.00, 1.00)",
        "score_expr": "p.p_xgb",
        "w_lgb": 0.00,
        "w_cb": 0.00,
        "w_xgb": 1.00,
    },
    {
        "model": "Blend A",
        "weights": "(0.50, 0.30, 0.20)",
        "score_expr": "0.50 * p.p_lgb + 0.30 * p.p_cb + 0.20 * p.p_xgb",
        "w_lgb": 0.50,
        "w_cb": 0.30,
        "w_xgb": 0.20,
    },
    {
        "model": "Blend B",
        "weights": "(0.60, 0.25, 0.15)",
        "score_expr": "0.60 * p.p_lgb + 0.25 * p.p_cb + 0.15 * p.p_xgb",
        "w_lgb": 0.60,
        "w_cb": 0.25,
        "w_xgb": 0.15,
    },
    {
        "model": "Blend C",
        "weights": "(0.65, 0.20, 0.15)",
        "score_expr": "0.65 * p.p_lgb + 0.20 * p.p_cb + 0.15 * p.p_xgb",
        "w_lgb": 0.65,
        "w_cb": 0.20,
        "w_xgb": 0.15,
    },
    {
        "model": "Blend D",
        "weights": "(0.70, 0.20, 0.10)",
        "score_expr": "0.70 * p.p_lgb + 0.20 * p.p_cb + 0.10 * p.p_xgb",
        "w_lgb": 0.70,
        "w_cb": 0.20,
        "w_xgb": 0.10,
    },
    {
        "model": "Blend E",
        "weights": "(0.55, 0.20, 0.25)",
        "score_expr": "0.55 * p.p_lgb + 0.20 * p.p_cb + 0.25 * p.p_xgb",
        "w_lgb": 0.55,
        "w_cb": 0.20,
        "w_xgb": 0.25,
    },
]


def sql_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def evaluate_threshold_grid(
    con: duckdb.DuckDBPyConnection, thresholds: list[float]
) -> list[dict]:
    values_sql = ", ".join(f"({x:.6f})" for x in sorted(set(thresholds)))
    query = f"""
        WITH threshold_list(threshold) AS (
            VALUES {values_sql}
        ),
        pred AS (
            SELECT
                t.threshold,
                b.source1_entity_id,
                SUM(b.tp)::BIGINT AS tp,
                COUNT(*)::BIGINT AS predicted_count
            FROM threshold_list t
            INNER JOIN scored b
              ON b.score >= t.threshold
            GROUP BY t.threshold, b.source1_entity_id
        ),
        joined AS (
            SELECT
                t.threshold,
                e.source1_entity_id,
                e.true_count,
                COALESCE(p.tp, 0)::BIGINT AS tp,
                COALESCE(p.predicted_count, 0)::BIGINT AS predicted_count
            FROM threshold_list t
            CROSS JOIN entity_gt e
            LEFT JOIN pred p
              ON p.threshold = t.threshold
             AND p.source1_entity_id = e.source1_entity_id
        ),
        per_entity AS (
            SELECT
                threshold,
                source1_entity_id,
                true_count,
                tp,
                predicted_count,
                CASE
                    WHEN predicted_count = 0 AND true_count = 0 THEN 1.0
                    WHEN predicted_count = 0 OR true_count = 0 THEN 0.0
                    ELSE
                        (1.25 * tp)::DOUBLE
                        / (1.25 * tp
                           + 0.25 * (predicted_count - tp)
                           + (true_count - tp))
                END AS f05
            FROM joined
        )
        SELECT
            threshold::DOUBLE AS threshold,
            COUNT(*)::BIGINT AS validation_s1_count,
            SUM(tp)::BIGINT AS tp,
            SUM(predicted_count - tp)::BIGINT AS fp,
            SUM(true_count - tp)::BIGINT AS fn,
            SUM(predicted_count)::BIGINT AS predicted_pairs,
            SUM(true_count)::BIGINT AS true_pairs,
            AVG(f05)::DOUBLE AS macro_f05,
            AVG(CASE WHEN predicted_count > 0
                     THEN tp::DOUBLE / predicted_count ELSE 0.0 END)::DOUBLE AS macro_precision,
            AVG(CASE WHEN true_count > 0
                     THEN tp::DOUBLE / true_count ELSE 0.0 END)::DOUBLE AS macro_recall
        FROM per_entity
        GROUP BY threshold
        ORDER BY macro_f05 DESC
    """
    rows = con.execute(query).fetchall()
    cols = [desc[0] for desc in con.description]
    return [dict(zip(cols, row)) for row in rows]


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Tune ensemble weights and evaluate Macro F0.5 on validation split."
    )
    ap.add_argument(
        "--pred",
        type=Path,
        default=PRED_PATH,
        help="Path to scored_valid_pairs_phase3_fixed.parquet",
    )
    ap.add_argument(
        "--valid",
        type=Path,
        default=VALID_PATH,
        help="Path to valid_split.parquet",
    )
    ap.add_argument(
        "--gt",
        type=Path,
        default=GT_PATH,
        help="Path to ground_truth_pairs.parquet",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=OUT_RESULTS,
        help="Path to save ensemble_weight_results.parquet",
    )
    ap.add_argument(
        "--threads",
        type=int,
        default=8,
        help="DuckDB threads (default 8)",
    )
    ap.add_argument(
        "--memory-limit",
        type=str,
        default="6GB",
        help="DuckDB memory limit (default 6GB)",
    )
    args = ap.parse_args()

    # 1. Verification of inputs
    for p in (args.pred, args.valid, args.gt):
        if not p.exists():
            raise FileNotFoundError(f"Required input file missing: {p}")

    p_sql = sql_path(args.pred)
    v_sql = sql_path(args.valid)
    g_sql = sql_path(args.gt)

    print("=" * 80)
    print("ENSEMBLE WEIGHT & THRESHOLD TUNING EXPERIMENT")
    print("=" * 80)
    print(f"Prediction file : {args.pred}")
    print(f"Validation split: {args.valid}")
    print(f"Ground truth    : {args.gt}")
    print(f"DuckDB threads  : {args.threads}")
    print(f"Memory limit    : {args.memory_limit}")
    print(f"Protected Base  : Macro F0.5 = {PROTECTED_BASELINE_F05:.4f}")
    print("-" * 80)

    # 2. Initialize DuckDB
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = false")

    # 3. Validate required columns in prediction parquet
    pred_cols = {
        r[0] for r in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{p_sql}')").fetchall()
    }
    missing_cols = REQUIRED_COLUMNS - pred_cols
    if missing_cols:
        raise ValueError(
            f"Prediction parquet missing required columns: {sorted(missing_cols)}"
        )

    # 4. Check for duplicate pair groups (source1 + matched_id + matched_source)
    print("Verifying candidate pair identity uniqueness...")
    duplicate_groups = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT source1_entity_id, matched_entity_id, matched_source
            FROM read_parquet('{p_sql}')
            GROUP BY 1,2,3
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if duplicate_groups > 0:
        raise ValueError(
            f"Duplicate prediction pair groups detected using full pair identity: {duplicate_groups:,}"
        )
    print("Pair identity uniqueness confirmed: 0 duplicate groups.")

    # 5. Populate reference tables (valid_s1, gt_valid, entity_gt)
    print("Pre-aggregating validation population and ground truth...")
    t_start = time.time()
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE valid_s1 AS
        SELECT DISTINCT source1_entity_id
        FROM read_parquet('{v_sql}')
    """)
    s1_count = con.execute("SELECT COUNT(*) FROM valid_s1").fetchone()[0]

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_valid AS
        SELECT DISTINCT
            g.source1_entity_id,
            g.matched_entity_id,
            g.matched_source
        FROM read_parquet('{g_sql}') g
        INNER JOIN valid_s1 s USING (source1_entity_id)
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE entity_gt AS
        SELECT
            s.source1_entity_id,
            COUNT(g.matched_entity_id)::BIGINT AS true_count
        FROM valid_s1 s
        LEFT JOIN gt_valid g USING (source1_entity_id)
        GROUP BY s.source1_entity_id
    """)
    print(f"Validation S1 population: {s1_count:,} entities ready in {time.time() - t_start:.2f}s.")

    # 6. Evaluation loop
    results: list[dict] = []

    print("-" * 80)
    print(f"Evaluating {len(MODELS_AND_BLENDS)} candidate models / blends...")

    for i, cfg in enumerate(MODELS_AND_BLENDS, 1):
        m_name = cfg["model"]
        weights_str = cfg["weights"]
        score_expr = cfg["score_expr"]
        print(f"\n[{i}/{len(MODELS_AND_BLENDS)}] {m_name:<15} Weights: {weights_str} ...", flush=True)
        t_eval = time.time()

        # Materialize scored temp table for current model/blend
        con.execute(f"""
            CREATE OR REPLACE TEMP TABLE scored AS
            SELECT
                p.source1_entity_id,
                p.matched_entity_id,
                p.matched_source,
                CAST({score_expr} AS DOUBLE) AS score,
                CASE WHEN g.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS tp
            FROM read_parquet('{p_sql}') p
            INNER JOIN valid_s1 s USING (source1_entity_id)
            LEFT JOIN gt_valid g
              ON p.source1_entity_id = g.source1_entity_id
             AND p.matched_entity_id = g.matched_entity_id
             AND p.matched_source = g.matched_source
        """)

        # Coarse threshold search
        coarse_res = evaluate_threshold_grid(con, COARSE_THRESHOLDS)
        best_row = coarse_res[0]

        # Fine-grained local search around best coarse threshold
        best_th = float(best_row["threshold"])
        fine_candidates = [
            round(best_th - 0.02, 4),
            round(best_th - 0.01, 4),
            round(best_th + 0.01, 4),
            round(best_th + 0.02, 4),
        ]
        fine_candidates = [
            t for t in fine_candidates if 0.0 < t < 1.0 and t not in COARSE_THRESHOLDS
        ]
        if fine_candidates:
            fine_res = evaluate_threshold_grid(con, fine_candidates)
            if fine_res and fine_res[0]["macro_f05"] > best_row["macro_f05"]:
                best_row = fine_res[0]

        elapsed = time.time() - t_eval
        print(
            f"   -> Best Threshold: {best_row['threshold']:.4f} | "
            f"Macro F0.5: {best_row['macro_f05']:.6f} | "
            f"Precision: {best_row['macro_precision']:.4f} | "
            f"Recall: {best_row['macro_recall']:.4f} ({elapsed:.1f}s)",
            flush=True,
        )

        results.append(
            {
                "model": m_name,
                "weights": weights_str,
                "best_threshold": float(best_row["threshold"]),
                "macro_f05": float(best_row["macro_f05"]),
                "macro_precision": float(best_row["macro_precision"]),
                "macro_recall": float(best_row["macro_recall"]),
                "tp": int(best_row["tp"]),
                "fp": int(best_row["fp"]),
                "fn": int(best_row["fn"]),
                "predicted_pairs": int(best_row["predicted_pairs"]),
                "w_lgb": float(cfg["w_lgb"]),
                "w_cb": float(cfg["w_cb"]),
                "w_xgb": float(cfg["w_xgb"]),
            }
        )

    # 7. Sort results by Macro F0.5 descending
    results.sort(key=lambda x: x["macro_f05"], reverse=True)

    # 8. Print formatted comparison table
    print("\n" + "=" * 125)
    header = (
        f"{'MODEL':<16} {'WEIGHTS':<20} {'BEST_THRESHOLD':<15} {'MACRO_F0.5':<12} "
        f"{'MACRO_PRECISION':<17} {'MACRO_RECALL':<14} {'TP':<10} {'FP':<10} {'FN':<10} {'PREDICTED_PAIRS':<15}"
    )
    print(header)
    print("-" * 125)
    for r in results:
        row_str = (
            f"{r['model']:<16} "
            f"{r['weights']:<20} "
            f"{r['best_threshold']:<15.4f} "
            f"{r['macro_f05']:<12.6f} "
            f"{r['macro_precision']:<17.6f} "
            f"{r['macro_recall']:<14.6f} "
            f"{r['tp']:<10d} "
            f"{r['fp']:<10d} "
            f"{r['fn']:<10d} "
            f"{r['predicted_pairs']:<15d}"
        )
        print(row_str)
    print("=" * 125)

    # 9. Gate check vs Protected Baseline
    best_exp = results[0]
    best_f05 = best_exp["macro_f05"]
    delta = best_f05 - PROTECTED_BASELINE_F05
    promote = "YES" if best_f05 >= PROTECTED_BASELINE_F05 else "NO"

    print("\n" + "=" * 50)
    print("BASELINE COMPARISON & PROMOTION GATE")
    print("=" * 50)
    print(f"PROTECTED BASELINE    : {PROTECTED_BASELINE_F05:.4f}")
    print(f"BEST EXPERIMENTAL F0.5: {best_f05:.6f} ({best_exp['model']} @ threshold {best_exp['best_threshold']:.4f})")
    print(f"DELTA VS BASELINE     : {delta:+.6f}")
    print(f"PROMOTE               : {promote}")
    print("=" * 50)

    if promote == "NO":
        print("\n>>> ENSEMBLE REJECTED — KEEP PROVEN FINAL LIGHTGBM. <<<")
    else:
        print("\n>>> ENSEMBLE EXCEEDS BASELINE — READY FOR REVIEW. <<<")

    # 10. Save results to compact parquet
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_sql = sql_path(args.out)

    # Populate temporary table for export
    con.execute("""
        CREATE OR REPLACE TEMP TABLE final_summary_results (
            model VARCHAR,
            weights VARCHAR,
            best_threshold DOUBLE,
            macro_f05 DOUBLE,
            macro_precision DOUBLE,
            macro_recall DOUBLE,
            tp BIGINT,
            fp BIGINT,
            fn BIGINT,
            predicted_pairs BIGINT,
            w_lgb DOUBLE,
            w_cb DOUBLE,
            w_xgb DOUBLE
        )
    """)
    for r in results:
        con.execute(
            """
            INSERT INTO final_summary_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                r["model"],
                r["weights"],
                r["best_threshold"],
                r["macro_f05"],
                r["macro_precision"],
                r["macro_recall"],
                r["tp"],
                r["fp"],
                r["fn"],
                r["predicted_pairs"],
                r["w_lgb"],
                r["w_cb"],
                r["w_xgb"],
            ],
        )

    con.execute(f"""
        COPY (SELECT * FROM final_summary_results ORDER BY macro_f05 DESC)
        TO '{out_sql}' (FORMAT PARQUET)
    """)
    print(f"\nSaved compact results table to: {args.out}")
    con.close()


if __name__ == "__main__":
    main()

from __future__ import annotations

"""
Adaptive Decision Post-Processing Experiment for Amazon ML Challenge 2026.
Evaluates adaptive thresholding rules:
    score >= max(T_abs, alpha * entity_max_score)
along with cardinality-safe Variants A and B on valid_predictions.parquet.

PROTECTED BASELINE:
    Macro F0.5 ≈ 0.765690 (at static threshold 0.585)
PROMOTION GATE:
    Macro F0.5 MUST be strictly greater than 0.765900.
"""

import argparse
import json
import os
from pathlib import Path
import sys
import time
import duckdb

ROOT = Path(__file__).resolve().parents[2]
FEATURE_DIR = ROOT / "artifacts" / "features" / "final_train"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
MODEL_DIR = ROOT / "artifacts" / "models"
TMP_DIR = BLOCKING_DIR / "duckdb_tmp"

PRED_PATH = FEATURE_DIR / "valid_predictions.parquet"
VALID_PATH = FEATURE_DIR / "valid_split.parquet"
GT_PATH = BLOCKING_DIR / "ground_truth_pairs.parquet"

OUT_RESULTS = MODEL_DIR / "adaptive_decision_results.parquet"
OUT_CONFIG = MODEL_DIR / "adaptive_decision_config.json"

PROTECTED_BASELINE_F05 = 0.765900
EXPECTED_BASELINE_F05 = 0.765690
BASELINE_THRESHOLD = 0.585

T_ABS_GRID = [0.55, 0.575, 0.585, 0.60]
ALPHA_GRID = [0.90, 0.92, 0.94, 0.95, 0.96, 0.97, 0.98, 0.99]
CARD_BUCKETS = ["0", "1", "2", "3", "4", "5", "6+"]

REQUIRED_COLUMNS = {
    "source1_entity_id",
    "matched_entity_id",
    "matched_source",
    "score",
}


def sql_path(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/")


def evaluate_condition(
    con: duckdb.DuckDBPyConnection, cond_sql: str
) -> tuple[dict, list[dict]]:
    """
    Evaluates a pair selection condition against s1_full and pred in DuckDB.
    Returns:
        overall_metrics: dict with TP, FP, FN, predicted_pairs, macro_f05, macro_precision, macro_recall
        card_metrics: list of dicts with cardinality bucket statistics
    """
    query = f"""
        WITH predicted AS (
            SELECT
                p.source1_entity_id,
                COUNT(*)::BIGINT AS predicted_count,
                SUM(p.is_tp)::BIGINT AS tp
            FROM pred p
            INNER JOIN s1_full s USING (source1_entity_id)
            WHERE {cond_sql}
            GROUP BY p.source1_entity_id
        ),
        entity_scores AS (
            SELECT
                s.source1_entity_id,
                s.true_count,
                s.card_bucket,
                COALESCE(p.predicted_count, 0)::BIGINT AS predicted_count,
                COALESCE(p.tp, 0)::BIGINT AS tp
            FROM s1_full s
            LEFT JOIN predicted p USING (source1_entity_id)
        ),
        scored AS (
            SELECT
                source1_entity_id,
                card_bucket,
                predicted_count,
                true_count,
                tp,
                predicted_count - tp AS fp,
                true_count - tp AS fn,
                CASE
                    WHEN predicted_count = 0 AND true_count = 0 THEN 1.0
                    WHEN 0.25 * (CASE WHEN predicted_count > 0 THEN tp::DOUBLE / predicted_count ELSE 0.0 END)
                         + (CASE WHEN true_count > 0 THEN tp::DOUBLE / true_count ELSE 0.0 END) = 0 THEN 0.0
                    ELSE 1.25 * (CASE WHEN predicted_count > 0 THEN tp::DOUBLE / predicted_count ELSE 0.0 END)
                              * (CASE WHEN true_count > 0 THEN tp::DOUBLE / true_count ELSE 0.0 END)
                         / (0.25 * (CASE WHEN predicted_count > 0 THEN tp::DOUBLE / predicted_count ELSE 0.0 END)
                            + (CASE WHEN true_count > 0 THEN tp::DOUBLE / true_count ELSE 0.0 END))
                END AS f05,
                CASE WHEN predicted_count > 0 THEN tp::DOUBLE / predicted_count ELSE 0.0 END AS precision,
                CASE WHEN true_count > 0 THEN tp::DOUBLE / true_count ELSE 0.0 END AS recall
            FROM entity_scores
        )
        SELECT
            SUM(tp)::BIGINT AS tp,
            SUM(fp)::BIGINT AS fp,
            SUM(fn)::BIGINT AS fn,
            SUM(predicted_count)::BIGINT AS predicted_pairs,
            AVG(f05)::DOUBLE AS macro_f05,
            AVG(precision)::DOUBLE AS macro_precision,
            AVG(recall)::DOUBLE AS macro_recall
        FROM scored
    """
    row = con.execute(query).fetchone()
    overall = {
        "tp": int(row[0]),
        "fp": int(row[1]),
        "fn": int(row[2]),
        "predicted_pairs": int(row[3]),
        "macro_f05": float(row[4]),
        "macro_precision": float(row[5]),
        "macro_recall": float(row[6]),
    }

    card_query = f"""
        WITH predicted AS (
            SELECT
                p.source1_entity_id,
                COUNT(*)::BIGINT AS predicted_count,
                SUM(p.is_tp)::BIGINT AS tp
            FROM pred p
            INNER JOIN s1_full s USING (source1_entity_id)
            WHERE {cond_sql}
            GROUP BY p.source1_entity_id
        ),
        entity_scores AS (
            SELECT
                s.source1_entity_id,
                s.true_count,
                s.card_bucket,
                COALESCE(p.predicted_count, 0)::BIGINT AS predicted_count,
                COALESCE(p.tp, 0)::BIGINT AS tp
            FROM s1_full s
            LEFT JOIN predicted p USING (source1_entity_id)
        ),
        scored AS (
            SELECT
                source1_entity_id,
                card_bucket,
                predicted_count,
                true_count,
                tp,
                CASE
                    WHEN predicted_count = 0 AND true_count = 0 THEN 1.0
                    WHEN 0.25 * (CASE WHEN predicted_count > 0 THEN tp::DOUBLE / predicted_count ELSE 0.0 END)
                         + (CASE WHEN true_count > 0 THEN tp::DOUBLE / true_count ELSE 0.0 END) = 0 THEN 0.0
                    ELSE 1.25 * (CASE WHEN predicted_count > 0 THEN tp::DOUBLE / predicted_count ELSE 0.0 END)
                              * (CASE WHEN true_count > 0 THEN tp::DOUBLE / true_count ELSE 0.0 END)
                         / (0.25 * (CASE WHEN predicted_count > 0 THEN tp::DOUBLE / predicted_count ELSE 0.0 END)
                            + (CASE WHEN true_count > 0 THEN tp::DOUBLE / true_count ELSE 0.0 END))
                END AS f05
            FROM entity_scores
        )
        SELECT
            card_bucket,
            COUNT(*)::BIGINT AS entity_count,
            AVG(predicted_count)::DOUBLE AS avg_predicted_matches,
            AVG(f05)::DOUBLE AS macro_f05
        FROM scored
        GROUP BY card_bucket
        ORDER BY
            CASE card_bucket
                WHEN '0' THEN 0
                WHEN '1' THEN 1
                WHEN '2' THEN 2
                WHEN '3' THEN 3
                WHEN '4' THEN 4
                WHEN '5' THEN 5
                ELSE 6
            END
    """
    card_rows = con.execute(card_query).fetchall()
    card_list = [
        {
            "card_bucket": r[0],
            "entity_count": int(r[1]),
            "avg_predicted_matches": float(r[2]),
            "macro_f05": float(r[3]),
        }
        for r in card_rows
    ]

    return overall, card_list


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Evaluate adaptive decision post-processing on validation predictions."
    )
    ap.add_argument(
        "--pred",
        type=Path,
        default=PRED_PATH,
        help="Path to valid_predictions.parquet",
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
        help="Path to save adaptive_decision_results.parquet",
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
    print("ADAPTIVE DECISION POST-PROCESSING EXPERIMENT")
    print("=" * 80)
    print(f"Prediction file : {args.pred}")
    print(f"Validation split: {args.valid}")
    print(f"Ground truth    : {args.gt}")
    print(f"DuckDB threads  : {args.threads}")
    print(f"Memory limit    : {args.memory_limit}")
    print(f"Protected Base  : Macro F0.5 = {PROTECTED_BASELINE_F05:.6f}")
    print("-" * 80)

    # 2. Initialize DuckDB with disk spilling
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect()
    con.execute(f"SET threads = {args.threads}")
    con.execute(f"SET memory_limit = '{args.memory_limit}'")
    con.execute("SET preserve_insertion_order = false")
    con.execute(f"SET temp_directory = '{sql_path(TMP_DIR)}'")

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

    # 5. Populate reference tables (valid_s1, gt_valid, pred, entity_gt, entity_cand, s1_full)
    print("Pre-aggregating validation population and entity statistics...")
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

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE pred AS
        SELECT
            p.source1_entity_id,
            p.matched_entity_id,
            p.matched_source,
            CAST(p.score AS DOUBLE) AS score,
            CASE WHEN g.source1_entity_id IS NOT NULL THEN 1 ELSE 0 END AS is_tp
        FROM read_parquet('{p_sql}') p
        INNER JOIN valid_s1 s USING (source1_entity_id)
        LEFT JOIN gt_valid g
          ON p.source1_entity_id = g.source1_entity_id
         AND p.matched_entity_id = g.matched_entity_id
         AND p.matched_source = g.matched_source
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

    con.execute("""
        CREATE OR REPLACE TEMP TABLE entity_cand AS
        SELECT
            source1_entity_id,
            MAX(score) AS max_score,
            COUNT(*)::INT AS cand_count,
            SUM(CASE WHEN score >= 0.585 THEN 1 ELSE 0 END)::INT AS base_pred_count
        FROM pred
        GROUP BY source1_entity_id
    """)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE s1_full AS
        SELECT
            e.source1_entity_id,
            e.true_count,
            COALESCE(c.max_score, 0.0) AS max_score,
            COALESCE(c.cand_count, 0) AS cand_count,
            COALESCE(c.base_pred_count, 0) AS base_pred_count,
            CASE
                WHEN e.true_count = 0 THEN '0'
                WHEN e.true_count = 1 THEN '1'
                WHEN e.true_count = 2 THEN '2'
                WHEN e.true_count = 3 THEN '3'
                WHEN e.true_count = 4 THEN '4'
                WHEN e.true_count = 5 THEN '5'
                ELSE '6+'
            END AS card_bucket
        FROM entity_gt e
        LEFT JOIN entity_cand c USING (source1_entity_id)
    """)
    print(f"Validation S1 population: {s1_count:,} entities ready in {time.time() - t_start:.2f}s.")

    # -------------------------------------------------------------------------
    # STEP 1: REPRODUCE BASELINE
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 1: REPRODUCING PRODUCTION BASELINE (Threshold = 0.585)")
    print("=" * 80)
    base_overall, base_card = evaluate_condition(con, f"p.score >= {BASELINE_THRESHOLD}")

    print(f"Macro F0.5       : {base_overall['macro_f05']:.6f}")
    print(f"Macro Precision  : {base_overall['macro_precision']:.6f}")
    print(f"Macro Recall     : {base_overall['macro_recall']:.6f}")
    print(f"TP               : {base_overall['tp']:,}")
    print(f"FP               : {base_overall['fp']:,}")
    print(f"FN               : {base_overall['fn']:,}")
    print(f"Predicted pairs  : {base_overall['predicted_pairs']:,}")

    diff = abs(base_overall["macro_f05"] - EXPECTED_BASELINE_F05)
    if diff > 0.001:
        print(f"\n[FATAL ERROR] Baseline reproduction discrepancy detected!")
        print(f"Expected ≈ {EXPECTED_BASELINE_F05:.6f}, got {base_overall['macro_f05']:.6f} (diff = {diff:.6f})")
        print("STOPPING as requested.")
        sys.exit(1)
    else:
        print(f"[SUCCESS] Baseline reproduced exactly within tolerance (diff = {diff:.6f}).")

    # -------------------------------------------------------------------------
    # STEP 2, 3, 4: EVALUATE TARGETED GRID & CARDINALITY-SAFE VARIANTS
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("STEP 2-4: EVALUATING ADAPTIVE DECISION GRID & VARIANTS")
    print("=" * 80)

    results: list[dict] = []
    # Add baseline row
    results.append(
        {
            "variant": "Baseline (Static)",
            "t_abs": BASELINE_THRESHOLD,
            "alpha": 0.0,
            "macro_f05": base_overall["macro_f05"],
            "macro_precision": base_overall["macro_precision"],
            "macro_recall": base_overall["macro_recall"],
            "tp": base_overall["tp"],
            "fp": base_overall["fp"],
            "fn": base_overall["fn"],
            "predicted_pairs": base_overall["predicted_pairs"],
        }
    )

    variants = [
        (
            "Standard",
            lambda t, a: f"p.score >= GREATEST({t:.6f}, {a:.6f} * s.max_score)",
        ),
        (
            "Variant A",
            lambda t, a: f"p.score >= (CASE WHEN s.base_pred_count >= 2 THEN GREATEST({t:.6f}, {a:.6f} * s.max_score) ELSE {BASELINE_THRESHOLD:.6f} END)",
        ),
        (
            "Variant B",
            lambda t, a: f"p.score >= (CASE WHEN s.cand_count >= 3 THEN GREATEST({t:.6f}, {a:.6f} * s.max_score) ELSE {BASELINE_THRESHOLD:.6f} END)",
        ),
    ]

    total_configs = len(variants) * len(T_ABS_GRID) * len(ALPHA_GRID)
    cfg_idx = 0
    t_grid_start = time.time()

    for v_name, cond_fn in variants:
        for t_abs in T_ABS_GRID:
            for alpha in ALPHA_GRID:
                cfg_idx += 1
                cond_sql = cond_fn(t_abs, alpha)
                overall, _ = evaluate_condition(con, cond_sql)

                results.append(
                    {
                        "variant": v_name,
                        "t_abs": t_abs,
                        "alpha": alpha,
                        "macro_f05": overall["macro_f05"],
                        "macro_precision": overall["macro_precision"],
                        "macro_recall": overall["macro_recall"],
                        "tp": overall["tp"],
                        "fp": overall["fp"],
                        "fn": overall["fn"],
                        "predicted_pairs": overall["predicted_pairs"],
                    }
                )

                if cfg_idx % 16 == 0 or cfg_idx == total_configs:
                    print(
                        f"Evaluated [{cfg_idx:2d}/{total_configs}] configs... "
                        f"Latest: {v_name} (T_abs={t_abs:.3f}, alpha={alpha:.2f}) -> Macro F0.5: {overall['macro_f05']:.6f}"
                    )

    print(f"Grid evaluation finished in {time.time() - t_grid_start:.2f}s.")

    # -------------------------------------------------------------------------
    # STEP 5: CARDINALITY BREAKDOWN & METRICS
    # -------------------------------------------------------------------------
    # Sort all results descending by Macro F0.5
    results.sort(key=lambda x: x["macro_f05"], reverse=True)

    print("\n" + "=" * 120)
    print("TOP 10 CONFIGURATIONS (Sorted by Macro F0.5 Descending)")
    print("=" * 120)
    header = (
        f"{'VARIANT':<18} {'T_ABS':<8} {'ALPHA':<8} {'MACRO_F0.5':<12} "
        f"{'MACRO_PRECISION':<17} {'MACRO_RECALL':<14} {'TP':<10} {'FP':<10} {'FN':<10} {'PRED_PAIRS':<12}"
    )
    print(header)
    print("-" * 120)
    for r in results[:10]:
        print(
            f"{r['variant']:<18} {r['t_abs']:<8.4f} {r['alpha']:<8.4f} {r['macro_f05']:<12.6f} "
            f"{r['macro_precision']:<17.6f} {r['macro_recall']:<14.6f} "
            f"{r['tp']:<10d} {r['fp']:<10d} {r['fn']:<10d} {r['predicted_pairs']:<12d}"
        )
    print("=" * 120)

    # Find best adaptive configuration (excluding static baseline)
    adaptive_configs = [r for r in results if r["variant"] != "Baseline (Static)"]
    best_adaptive = adaptive_configs[0]

    # Re-evaluate best adaptive configuration to get its cardinality breakdown
    if best_adaptive["variant"] == "Standard":
        best_cond = f"p.score >= GREATEST({best_adaptive['t_abs']:.6f}, {best_adaptive['alpha']:.6f} * s.max_score)"
    elif best_adaptive["variant"] == "Variant A":
        best_cond = f"p.score >= (CASE WHEN s.base_pred_count >= 2 THEN GREATEST({best_adaptive['t_abs']:.6f}, {best_adaptive['alpha']:.6f} * s.max_score) ELSE {BASELINE_THRESHOLD:.6f} END)"
    else:
        best_cond = f"p.score >= (CASE WHEN s.cand_count >= 3 THEN GREATEST({best_adaptive['t_abs']:.6f}, {best_adaptive['alpha']:.6f} * s.max_score) ELSE {BASELINE_THRESHOLD:.6f} END)"

    _, best_card = evaluate_condition(con, best_cond)

    print("\n" + "=" * 80)
    print("PERFORMANCE BY TRUE-MATCH CARDINALITY (Baseline vs Best Adaptive)")
    print("=" * 80)
    print(f"{'BUCKET':<8} {'ENTITIES':<12} {'BASE AVG PRED':<16} {'BASE F0.5':<14} {'ADAPT AVG PRED':<16} {'ADAPT F0.5':<12}")
    print("-" * 80)
    for b_row, a_row in zip(base_card, best_card):
        print(
            f"{b_row['card_bucket']:<8} "
            f"{b_row['entity_count']:<12,d} "
            f"{b_row['avg_predicted_matches']:<16.4f} "
            f"{b_row['macro_f05']:<14.4f} "
            f"{a_row['avg_predicted_matches']:<16.4f} "
            f"{a_row['macro_f05']:<12.4f}"
        )
    print("=" * 80)

    # -------------------------------------------------------------------------
    # STEP 6: PROMOTION GATE
    # -------------------------------------------------------------------------
    best_f05 = best_adaptive["macro_f05"]
    delta = best_f05 - PROTECTED_BASELINE_F05
    promote = "YES" if best_f05 > PROTECTED_BASELINE_F05 else "NO"

    print("\n============================================================")
    print("ADAPTIVE DECISION EXPERIMENT")
    print("============================================================")
    print(f"PROTECTED BASELINE:")
    print(f"{PROTECTED_BASELINE_F05:.6f}")
    print()
    print(f"BEST ADAPTIVE:")
    print(f"{best_f05:.6f}")
    print()
    print(f"DELTA:")
    print(f"{delta:+.6f}")
    print()
    print(f"BEST T_ABS:")
    print(f"{best_adaptive['t_abs']:.4f}")
    print()
    print(f"BEST ALPHA:")
    print(f"{best_adaptive['alpha']:.4f}")
    print()
    print(f"BEST VARIANT:")
    print(f"{best_adaptive['variant']}")
    print()
    print(f"PROMOTE:")
    print(f"{promote}")
    print("============================================================\n")

    if promote == "NO":
        print("ADAPTIVE DECISION REJECTED.")
        print("KEEP STATIC THRESHOLD 0.585.")
    else:
        print("ADAPTIVE DECISION PASSES LOCAL PROMOTION GATE.")

    # -------------------------------------------------------------------------
    # STEP 7: OUTPUT ARTIFACTS
    # -------------------------------------------------------------------------
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_sql = sql_path(args.out)

    con.execute("""
        CREATE OR REPLACE TEMP TABLE final_results (
            variant VARCHAR,
            t_abs DOUBLE,
            alpha DOUBLE,
            macro_f05 DOUBLE,
            macro_precision DOUBLE,
            macro_recall DOUBLE,
            tp BIGINT,
            fp BIGINT,
            fn BIGINT,
            predicted_pairs BIGINT
        )
    """)
    for r in results:
        con.execute(
            """
            INSERT INTO final_results VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            [
                r["variant"],
                r["t_abs"],
                r["alpha"],
                r["macro_f05"],
                r["macro_precision"],
                r["macro_recall"],
                r["tp"],
                r["fp"],
                r["fn"],
                r["predicted_pairs"],
            ],
        )

    con.execute(f"""
        COPY (SELECT * FROM final_results ORDER BY macro_f05 DESC)
        TO '{out_sql}' (FORMAT PARQUET)
    """)
    print(f"\nSaved compact results table to: {args.out}")

    if promote == "YES":
        config_data = {
            "variant": best_adaptive["variant"],
            "t_abs": best_adaptive["t_abs"],
            "alpha": best_adaptive["alpha"],
            "macro_f05": best_adaptive["macro_f05"],
            "baseline_f05": PROTECTED_BASELINE_F05,
            "delta": delta,
        }
        with open(OUT_CONFIG, "w", encoding="utf-8") as f:
            json.dump(config_data, f, indent=2)
        print(f"Saved winning configuration to: {OUT_CONFIG}")

    con.close()


if __name__ == "__main__":
    main()

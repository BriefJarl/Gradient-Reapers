from __future__ import annotations

"""
Decision threshold optimization for Amazon ML Challenge 2026.

Macro F_0.5 is precision-heavy (penalizes false merges 2x more than missed links).
A higher decision threshold (typically 0.60 - 0.85) is usually optimal.
This module searches for the threshold that maximizes Macro F_0.5 on the validation split.
"""

import sys
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import duckdb
import numpy as np

from src.evaluation.metrics import evaluate_predictions_duckdb, sql_quote


def find_optimal_threshold(
    con: duckdb.DuckDBPyConnection,
    scored_pairs_path: Path,
    ground_truth_path: Path,
    s1_split_path: Path,
    thresholds: list[float] | None = None,
    output_json: Path | None = None,
) -> dict[str, object]:
    """
    Search over candidate probability thresholds to find the threshold maximizing Macro F_0.5.

    `scored_pairs_path` must have columns:
        source1_entity_id, matched_entity_id, probability
    """
    if thresholds is None:
        thresholds = [round(t, 2) for t in np.arange(0.30, 0.92, 0.05)]

    print("=" * 80)
    print("THRESHOLD OPTIMIZATION FOR MACRO F_0.5")
    print("=" * 80)
    print(f"Scored pairs : {scored_pairs_path}")
    print(f"Ground truth : {ground_truth_path}")
    print(f"Candidate thresholds: {thresholds}\n")

    scored_sql = sql_quote(scored_pairs_path)

    cols = {row[0] for row in con.execute(f"DESCRIBE SELECT * FROM read_parquet('{scored_sql}')").fetchall()}
    prob_col = "pred_prob" if "pred_prob" in cols else "probability"

    results = []
    best_threshold = 0.50
    best_f05 = -1.0
    best_metrics = {}

    for thresh in thresholds:
        pred_query = f"""
            SELECT source1_entity_id, matched_entity_id
            FROM read_parquet('{scored_sql}')
            WHERE {prob_col} >= {thresh}
        """

        metrics = evaluate_predictions_duckdb(
            con=con,
            predictions_query_or_table=pred_query,
            ground_truth_path=ground_truth_path,
            s1_split_path=s1_split_path,
        )

        f05 = metrics["macro_f05"]
        prec = metrics["macro_precision"]
        rec = metrics["macro_recall"]
        sing_acc = metrics["singleton_accuracy"]

        print(
            f"Threshold: {thresh:.2f} | "
            f"Macro F0.5: {f05:.6f} | "
            f"Precision: {prec:.6f} | "
            f"Recall: {rec:.6f} | "
            f"Singleton Acc: {sing_acc:.4f}"
        )

        record = {
            "threshold": thresh,
            "macro_f05": f05,
            "macro_precision": prec,
            "macro_recall": rec,
            "singleton_accuracy": sing_acc,
        }
        results.append(record)

        if f05 > best_f05:
            best_f05 = f05
            best_threshold = thresh
            best_metrics = metrics

    # Local fine-grained refinement (+- 0.04 in steps of 0.01)
    if thresholds is None or len(thresholds) >= 5:
        print("\nRunning fine-grained local refinement around", f"{best_threshold:.2f}...")
        fine_candidates = sorted(list({
            round(t, 2)
            for t in np.arange(max(0.10, best_threshold - 0.04), min(0.98, best_threshold + 0.045), 0.01)
        }))
        for thresh in fine_candidates:
            if any(r["threshold"] == thresh for r in results):
                continue
            pred_query = f"""
                SELECT source1_entity_id, matched_entity_id
                FROM read_parquet('{scored_sql}')
                WHERE {prob_col} >= {thresh}
            """
            metrics = evaluate_predictions_duckdb(
                con=con,
                predictions_query_or_table=pred_query,
                ground_truth_path=ground_truth_path,
                s1_split_path=s1_split_path,
            )
            f05 = metrics["macro_f05"]
            prec = metrics["macro_precision"]
            rec = metrics["macro_recall"]
            sing_acc = metrics["singleton_accuracy"]

            print(
                f"Fine Threshold: {thresh:.2f} | "
                f"Macro F0.5: {f05:.6f} | "
                f"Precision: {prec:.6f} | "
                f"Recall: {rec:.6f} | "
                f"Singleton Acc: {sing_acc:.4f}"
            )
            record = {
                "threshold": thresh,
                "macro_f05": f05,
                "macro_precision": prec,
                "macro_recall": rec,
                "singleton_accuracy": sing_acc,
            }
            results.append(record)

            if f05 > best_f05:
                best_f05 = f05
                best_threshold = thresh
                best_metrics = metrics

    print("\n" + "=" * 80)
    print(f"OPTIMAL THRESHOLD FOUND: {best_threshold:.2f}")
    print(f"BEST MACRO F_0.5:        {best_f05:.6f}")
    print(f"PRECISION:               {best_metrics.get('macro_precision', 0):.6f}")
    print(f"RECALL:                  {best_metrics.get('macro_recall', 0):.6f}")
    print("=" * 80)

    summary = {
        "best_threshold": best_threshold,
        "best_macro_f05": best_f05,
        "best_metrics": best_metrics,
        "all_evaluations": results,
    }

    if output_json:
        output_json.parent.mkdir(parents=True, exist_ok=True)
        with open(output_json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved threshold metadata to: {output_json}")

    return summary

from __future__ import annotations

"""
Amazon ML Challenge 2026: Phase 4 Country-Specific Calibration & Top-K Pruning.

Tunes:
1. tau_us: Optimal decision threshold for United States entities.
2. tau_india: Optimal decision threshold for India entities.
3. relative_margin (delta): Keeps matches within delta of entity's max probability.
4. top_k: Maximum allowable matches per S1 entity to avoid chain-name penalties.

Evaluates against the official Macro F_0.5 metric across all validation entities.
"""

import json
import sys
from pathlib import Path
import time
import duckdb

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import evaluate_predictions_duckdb

SCORED_VAL_PATH = ROOT / "artifacts" / "features" / "scored_valid_pairs_phase3.parquet"
GROUND_TRUTH = ROOT / "artifacts" / "blocking" / "ground_truth_pairs.parquet"
VALID_S1 = ROOT / "artifacts" / "features" / "valid_phase3_s1_entities.parquet"
S1_NORM = ROOT / "artifacts" / "normalized" / "train_s1.parquet"
OUTPUT_JSON = ROOT / "artifacts" / "models" / "phase4_calibration_params.json"


def calibrate_parameters(con: duckdb.DuckDBPyConnection) -> dict[str, object]:
    print("=" * 80)
    print("PHASE 4: COUNTRY-SPECIFIC & TOP-K CALIBRATION")
    print("=" * 80)

    scored_sql = str(SCORED_VAL_PATH).replace("\\", "/").replace("'", "''")
    s1_norm_sql = str(S1_NORM).replace("\\", "/").replace("'", "''")

    # Create temporary table with country tag
    print("Tagging scored validation pairs with country...")
    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE scored_val_tagged AS
        SELECT
            v.source1_entity_id,
            v.matched_entity_id,
            v.pred_prob,
            COALESCE(s1.country_norm, 'other') AS country_norm
        FROM read_parquet('{scored_sql}') v
        LEFT JOIN read_parquet('{s1_norm_sql}') s1 ON v.source1_entity_id = s1.entity_id;
    """)

    best_score = -1.0
    best_config = {}

    tau_us_grid = [0.86, 0.88, 0.90]
    tau_in_grid = [0.82, 0.84, 0.86]
    delta_grid = [0.05, 0.08]
    top_k_grid = [5, 8]

    for tau_us in tau_us_grid:
        for tau_in in tau_in_grid:
            for delta in delta_grid:
                for top_k in top_k_grid:
                    pred_query = f"""
                        WITH filtered AS (
                            SELECT
                                source1_entity_id,
                                matched_entity_id,
                                pred_prob,
                                MAX(pred_prob) OVER (PARTITION BY source1_entity_id) AS max_p,
                                ROW_NUMBER() OVER (PARTITION BY source1_entity_id ORDER BY pred_prob DESC) AS rank
                            FROM scored_val_tagged
                            WHERE (country_norm = 'us' AND pred_prob >= {tau_us})
                               OR (country_norm = 'india' AND pred_prob >= {tau_in})
                               OR (country_norm NOT IN ('us', 'india') AND pred_prob >= {tau_us})
                        )
                        SELECT source1_entity_id, matched_entity_id
                        FROM filtered
                        WHERE pred_prob >= max_p - {delta}
                          AND rank <= {top_k}
                    """

                    metrics = evaluate_predictions_duckdb(
                        con=con,
                        predictions_query_or_table=pred_query,
                        ground_truth_path=GROUND_TRUTH,
                        s1_split_path=VALID_S1,
                    )

                    f05 = metrics["macro_f05"]
                    prec = metrics["macro_precision"]
                    rec = metrics["macro_recall"]

                    print(
                        f"US={tau_us:.2f} | IN={tau_in:.2f} | delta={delta:.2f} | top_k={top_k} => "
                        f"Macro F0.5: {f05:.6f} (P: {prec:.4f}, R: {rec:.4f})"
                    )

                    if f05 > best_score:
                        best_score = f05
                        best_config = {
                            "tau_us": tau_us,
                            "tau_india": tau_in,
                            "tau_other": tau_us,
                            "relative_margin": delta,
                            "top_k": top_k,
                            "macro_f05": f05,
                            "precision": prec,
                            "recall": rec,
                            "singleton_accuracy": metrics["singleton_accuracy"],
                        }

    print("\n" + "=" * 80)
    print("OPTIMAL PHASE 4 CALIBRATION CONFIGURATION")
    print("=" * 80)
    print(json.dumps(best_config, indent=2))

    with open(OUTPUT_JSON, "w") as f:
        json.dump(best_config, f, indent=2)
    print(f"\nConfiguration saved -> {OUTPUT_JSON}")

    return best_config


def main() -> None:
    con = duckdb.connect()
    con.execute("SET threads = 8")
    con.execute("SET memory_limit = '8GB'")
    try:
        calibrate_parameters(con)
    finally:
        con.close()


if __name__ == "__main__":
    main()

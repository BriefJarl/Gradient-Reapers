from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]

DEFAULT_PRED = ROOT / "artifacts" / "features" / "scored_valid_pairs_phase3_fixed.parquet"
DEFAULT_VALID = ROOT / "artifacts" / "features" / "final_train" / "valid_split.parquet"
DEFAULT_GT = ROOT / "artifacts" / "blocking" / "ground_truth_pairs.parquet"
DEFAULT_OUT = ROOT / "artifacts" / "models" / "ensemble_threshold_results.parquet"

THREADS = 8
MEMORY_LIMIT = "6GB"


def sql_path(p: Path) -> str:
    return str(p).replace("\\", "/").replace("'", "''")


def parse_thresholds(text: str) -> list[float]:
    vals = sorted({round(float(x.strip()), 6) for x in text.split(",") if x.strip()})
    if not vals or any(x < 0 or x > 1 for x in vals):
        raise ValueError("Thresholds must be non-empty and in [0,1].")
    return vals


def evaluate(con, pred, valid, gt, thresholds, score_col, delta):
    p, v, g = map(sql_path, (pred, valid, gt))

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE valid_s1 AS
        SELECT DISTINCT source1_entity_id
        FROM read_parquet('{v}')
    """)

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE gt_valid AS
        SELECT DISTINCT
            g.source1_entity_id,
            g.matched_entity_id,
            g.matched_source
        FROM read_parquet('{g}') g
        INNER JOIN valid_s1 s USING (source1_entity_id)
    """)

    cols = {r[0] for r in con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{p}')"
    ).fetchall()}

    required = {"source1_entity_id", "matched_entity_id", "matched_source", score_col}
    missing = required - cols
    if missing:
        raise ValueError(f"Prediction file missing required columns: {sorted(missing)}")

    # Correct pair identity: source1 + matched id + source.
    duplicate_groups = con.execute(f"""
        SELECT COUNT(*)
        FROM (
            SELECT source1_entity_id, matched_entity_id, matched_source
            FROM read_parquet('{p}')
            GROUP BY 1,2,3
            HAVING COUNT(*) > 1
        )
    """).fetchone()[0]

    if duplicate_groups:
        raise ValueError(
            f"Duplicate prediction pair groups using full pair identity: {duplicate_groups:,}"
        )

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE scored AS
        SELECT
            p.source1_entity_id,
            p.matched_entity_id,
            p.matched_source,
            CAST(p.{score_col} AS DOUBLE) AS score,
            CASE WHEN g.matched_entity_id IS NOT NULL THEN 1 ELSE 0 END AS tp
        FROM read_parquet('{p}') p
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

    values_sql = ", ".join(f"({x:.6f})" for x in thresholds)
    margin_sql = ""
    if delta is not None:
        margin_sql = f"AND score >= max_score - {float(delta):.8f}"

    con.execute(f"""
        CREATE OR REPLACE TEMP TABLE threshold_metrics AS
        WITH threshold_list(threshold) AS (
            VALUES {values_sql}
        ),
        base AS (
            SELECT
                s.*,
                MAX(s.score) OVER (PARTITION BY s.source1_entity_id) AS max_score
            FROM scored s
        ),
        pred AS (
            SELECT
                t.threshold,
                b.source1_entity_id,
                SUM(b.tp)::BIGINT AS tp,
                COUNT(*)::BIGINT AS predicted_count
            FROM threshold_list t
            INNER JOIN base b
              ON b.score >= t.threshold
              {margin_sql}
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
            threshold,
            COUNT(*)::BIGINT AS validation_s1_count,
            SUM(tp)::BIGINT AS tp,
            SUM(predicted_count - tp)::BIGINT AS fp,
            SUM(true_count - tp)::BIGINT AS fn,
            SUM(predicted_count)::BIGINT AS predicted_pairs,
            SUM(true_count)::BIGINT AS true_pairs,
            AVG(f05)::DOUBLE AS macro_f05,
            AVG(CASE WHEN predicted_count > 0
                     THEN tp::DOUBLE / predicted_count ELSE 0.0 END) AS macro_precision,
            AVG(CASE WHEN true_count > 0
                     THEN tp::DOUBLE / true_count ELSE 0.0 END) AS macro_recall
        FROM per_entity
        GROUP BY threshold
        ORDER BY threshold
    """)

    return con.execute("""
        SELECT threshold, validation_s1_count, tp, fp, fn,
               predicted_pairs, true_pairs, macro_f05,
               macro_precision, macro_recall
        FROM threshold_metrics
        ORDER BY threshold
    """).fetchall()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(DEFAULT_PRED))
    parser.add_argument("--valid-split", default=str(DEFAULT_VALID))
    parser.add_argument("--ground-truth", default=str(DEFAULT_GT))
    parser.add_argument("--score-col", default="pred_prob")
    parser.add_argument(
        "--thresholds",
        default="0.35,0.40,0.45,0.50,0.525,0.55,0.575,0.60,0.625,0.65,0.675,0.70,0.75,0.80"
    )
    parser.add_argument("--delta", type=float, default=None)
    parser.add_argument("--output", default=str(DEFAULT_OUT))
    args = parser.parse_args()

    pred, valid, gt, out = map(Path, (
        args.predictions, args.valid_split, args.ground_truth, args.output
    ))
    for p in (pred, valid, gt):
        if not p.exists():
            raise FileNotFoundError(p)

    thresholds = parse_thresholds(args.thresholds)

    print("=" * 92)
    print("AMAZON ML CHALLENGE 2026")
    print("ENSEMBLE ENTITY-LEVEL MACRO F0.5 EVALUATION")
    print("=" * 92)
    print(f"Predictions : {pred}")
    print(f"Validation  : {valid}")
    print(f"Ground truth: {gt}")
    print(f"Score col   : {args.score_col}")
    print(f"Thresholds  : {thresholds}")
    print(f"Margin delta: {args.delta}")
    print(f"DuckDB      : {THREADS} threads / {MEMORY_LIMIT}")

    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order = false")
    tmp = ROOT / "artifacts" / "blocking" / "duckdb_tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{sql_path(tmp)}'")

    try:
        rows = evaluate(con, pred, valid, gt, thresholds, args.score_col, args.delta)

        print("\nTHRESHOLD RESULTS")
        print("-" * 92)
        for r in rows:
            print(
                f"threshold={r[0]:.3f} F0.5={r[7]:.6f} "
                f"macroP={r[8]:.6f} macroR={r[9]:.6f} pred={r[5]:,}"
            )

        best = max(rows, key=lambda r: r[7])

        print("\n" + "=" * 92)
        print("BEST ENSEMBLE THRESHOLD")
        print("=" * 92)
        print(f"Threshold       : {best[0]:.6f}")
        print(f"Macro F0.5      : {best[7]:.6f}")
        print(f"Macro Precision : {best[8]:.6f}")
        print(f"Macro Recall    : {best[9]:.6f}")
        print(f"TP              : {best[2]:,}")
        print(f"FP              : {best[3]:,}")
        print(f"FN              : {best[4]:,}")
        print(f"Predicted pairs : {best[5]:,}")
        print(f"Validation S1   : {best[1]:,}")

        out.parent.mkdir(parents=True, exist_ok=True)
        if out.exists():
            out.unlink()
        con.execute(
            f"COPY threshold_metrics TO '{sql_path(out)}' "
            f"(FORMAT PARQUET, COMPRESSION ZSTD)"
        )
        print(f"Results         : {out}")
    finally:
        con.close()


if __name__ == "__main__":
    main()

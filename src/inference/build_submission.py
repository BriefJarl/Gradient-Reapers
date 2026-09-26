from __future__ import annotations

"""
Production submission builder.

Outputs:
  artifacts/submission/matching_results.tsv
  artifacts/submission/candidate_pairs.tsv

The official format requires exactly one Source-1 row in each file, including
empty ID lists for entities with no candidates / no selected matches.
"""

import argparse
import json
import os
from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[2]
NORMALIZED = ROOT / "artifacts" / "normalized"
BLOCKING = ROOT / "artifacts" / "blocking" / "final_test"
FEATURES = ROOT / "artifacts" / "features" / "final_test"
SUBMISSION = ROOT / "artifacts" / "submission"

S1 = NORMALIZED / "test_s1.parquet"
CANDIDATES = BLOCKING / "final_candidates.parquet"
PRED = FEATURES / "test_predictions.parquet"
THRESHOLD_JSON = ROOT / "artifacts" / "models" / "entity_threshold.json"

DEFAULT_THRESHOLD = 0.585
THREADS = int(os.environ.get("DUCKDB_THREADS", "8"))
MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY", "6GB")
TMP_DIR = ROOT / "artifacts" / "blocking" / "duckdb_tmp"


def qpath(p: Path) -> str:
    return str(p.resolve()).replace("\\", "/").replace("'", "''")


def configure(con):
    SUBMISSION.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{qpath(TMP_DIR)}'")


def threshold() -> float:
    if THRESHOLD_JSON.exists():
        obj = json.loads(THRESHOLD_JSON.read_text(encoding="utf-8"))
        return float(obj.get("best_threshold", DEFAULT_THRESHOLD))
    return DEFAULT_THRESHOLD


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=None)
    args = ap.parse_args()

    for p in (S1, CANDIDATES, PRED):
        if not p.exists():
            raise FileNotFoundError(p)

    th = threshold() if args.threshold is None else float(args.threshold)
    if not 0 <= th <= 1:
        raise ValueError("threshold must be in [0,1]")

    matching = SUBMISSION / "matching_results.tsv"
    candidate = SUBMISSION / "candidate_pairs.tsv"
    m_partial = SUBMISSION / "matching_results.partial.tsv"
    c_partial = SUBMISSION / "candidate_pairs.partial.tsv"

    for p in (matching, candidate, m_partial, c_partial):
        if p.exists():
            p.unlink()

    con = duckdb.connect()
    try:
        configure(con)

        print("=" * 88)
        print("BUILDING SUBMISSION TSVs")
        print("=" * 88)
        print(f"Threshold: {th:.6f}")

        # candidate_pairs.tsv: last candidate set actually scored by the model.
        # IDs are sorted deterministically within each S1.
        con.execute(f"""
            COPY (
                SELECT
                    s.entity_id AS source1_entity_id,
                    COALESCE(
                        c.candidate_entity_ids,
                        ''
                    ) AS candidate_entity_ids
                FROM read_parquet('{qpath(S1)}') s
                LEFT JOIN (
                    SELECT
                        source1_entity_id,
                        STRING_AGG(
                            matched_entity_id,
                            ',' ORDER BY
                                CASE WHEN matched_source='S2' THEN 0 ELSE 1 END,
                                matched_entity_id
                        ) AS candidate_entity_ids
                    FROM (
                        SELECT DISTINCT
                            source1_entity_id,
                            matched_entity_id,
                            matched_source
                        FROM read_parquet('{qpath(CANDIDATES)}')
                    ) d
                    GROUP BY source1_entity_id
                ) c
                  ON s.entity_id=c.source1_entity_id
            )
            TO '{qpath(c_partial)}'
            (HEADER, DELIMITER '\t', QUOTE '', ESCAPE '')
        """)
        c_partial.replace(candidate)
        print(f"candidate_pairs.tsv -> {candidate}")

        # matching_results.tsv: thresholded model decisions.
        # Keep every S1 entity, including empty match lists.
        con.execute(f"""
            COPY (
                SELECT
                    s.entity_id AS source1_entity_id,
                    COALESCE(m.matched_entity_ids, '') AS matched_entity_ids
                FROM read_parquet('{qpath(S1)}') s
                LEFT JOIN (
                    SELECT
                        source1_entity_id,
                        STRING_AGG(
                            matched_entity_id,
                            ',' ORDER BY
                                CASE WHEN matched_source='S2' THEN 0 ELSE 1 END,
                                matched_entity_id
                        ) AS matched_entity_ids
                    FROM (
                        SELECT DISTINCT
                            source1_entity_id,
                            matched_entity_id,
                            matched_source
                        FROM read_parquet('{qpath(PRED)}')
                        WHERE score >= {th:.9f}
                    ) d
                    GROUP BY source1_entity_id
                ) m
                  ON s.entity_id=m.source1_entity_id
            )
            TO '{qpath(m_partial)}'
            (HEADER, DELIMITER '\t', QUOTE '', ESCAPE '')
        """)
        m_partial.replace(matching)
        print(f"matching_results.tsv -> {matching}")

        print("=" * 88)
        print("SUBMISSION TSV BUILD PASSED")
        print(f"Threshold: {th:.6f}")
        print("=" * 88)
    finally:
        con.close()


if __name__ == "__main__":
    main()

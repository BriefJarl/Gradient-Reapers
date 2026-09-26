from __future__ import annotations

"""Memory-safe validator for the Amazon ML Challenge 2026 submission files.

Uses DuckDB for the large set operations instead of materialising millions of
IDs in Python dictionaries/sets.
"""

import argparse
from pathlib import Path
import duckdb

EXPECTED_S1 = 1_732_544


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def configure(con: duckdb.DuckDBPyConnection, tmp_dir: Path | None) -> None:
    con.execute("SET threads=8")
    con.execute("SET memory_limit='4GB'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=false")
    if tmp_dir:
        tmp_dir.mkdir(parents=True, exist_ok=True)
        con.execute(f"SET temp_directory='{sql_path(tmp_dir)}'")


def csv_relation(path: Path, column: str) -> str:
    # The generated TSVs contain exactly two unquoted tab-separated columns.
    return (
        f"read_csv('{sql_path(path)}', delim='\t', header=true, "
        f"quote='', escape='', columns={{'source1_entity_id':'VARCHAR', '{column}':'VARCHAR'}})"
    )


def test_ids(con, path: Path) -> None:
    cols = con.execute(
        f"DESCRIBE SELECT * FROM read_parquet('{sql_path(path)}')"
    ).fetchall()
    names = {r[0] for r in cols}
    if "entity_id" not in names:
        raise RuntimeError(f"Missing entity_id in {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default="artifacts/submission/matching_results.tsv")
    ap.add_argument("--candidate", default="artifacts/submission/candidate_pairs.tsv")
    ap.add_argument("--test-s1", default="artifacts/normalized/test_s1.parquet")
    ap.add_argument("--test-s2", default="artifacts/normalized/test_s2.parquet")
    ap.add_argument("--test-s3", default="artifacts/normalized/test_s3.parquet")
    ap.add_argument("--tmp-dir", default="artifacts/blocking/duckdb_tmp")
    args = ap.parse_args()

    matching = Path(args.matching)
    candidate = Path(args.candidate)
    s1 = Path(args.test_s1)
    s2 = Path(args.test_s2)
    s3 = Path(args.test_s3)
    tmp = Path(args.tmp_dir)

    for p in (matching, candidate, s1, s2, s3):
        if not p.exists():
            raise FileNotFoundError(p)

    con = duckdb.connect()
    try:
        configure(con, tmp)
        for p in (s1, s2, s3):
            test_ids(con, p)

        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW s1 AS
            SELECT entity_id FROM read_parquet('{sql_path(s1)}')
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW s2 AS
            SELECT entity_id FROM read_parquet('{sql_path(s2)}')
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW s3 AS
            SELECT entity_id FROM read_parquet('{sql_path(s3)}')
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW m AS
            SELECT row_number() OVER () AS row_id, *
            FROM {csv_relation(matching, 'matched_entity_ids')}
        """)
        con.execute(f"""
            CREATE OR REPLACE TEMP VIEW c AS
            SELECT row_number() OVER () AS row_id, *
            FROM {csv_relation(candidate, 'candidate_entity_ids')}
        """)

        s1_count = int(con.execute("SELECT COUNT(*) FROM s1").fetchone()[0])
        if s1_count != EXPECTED_S1:
            raise RuntimeError(f"TEST S1 count mismatch: {s1_count:,} != {EXPECTED_S1:,}")

        # Header / row-count / uniqueness / exact S1 coverage.
        m_count, m_dupes = con.execute("""
            SELECT COUNT(*), COUNT(*) - COUNT(DISTINCT source1_entity_id) FROM m
        """).fetchone()
        c_count, c_dupes = con.execute("""
            SELECT COUNT(*), COUNT(*) - COUNT(DISTINCT source1_entity_id) FROM c
        """).fetchone()
        if int(m_count) != s1_count or int(m_dupes) != 0:
            raise RuntimeError(f"matching_results row/duplicate failure: rows={m_count:,}, dupes={m_dupes:,}")
        if int(c_count) != s1_count or int(c_dupes) != 0:
            raise RuntimeError(f"candidate_pairs row/duplicate failure: rows={c_count:,}, dupes={c_dupes:,}")

        missing_m = int(con.execute("""
            SELECT COUNT(*) FROM s1
            WHERE entity_id NOT IN (SELECT source1_entity_id FROM m)
        """).fetchone()[0])
        missing_c = int(con.execute("""
            SELECT COUNT(*) FROM s1
            WHERE entity_id NOT IN (SELECT source1_entity_id FROM c)
        """).fetchone()[0])
        extra_m = int(con.execute("""
            SELECT COUNT(*) FROM m
            WHERE source1_entity_id NOT IN (SELECT entity_id FROM s1)
        """).fetchone()[0])
        extra_c = int(con.execute("""
            SELECT COUNT(*) FROM c
            WHERE source1_entity_id NOT IN (SELECT entity_id FROM s1)
        """).fetchone()[0])
        if any((missing_m, missing_c, extra_m, extra_c)):
            raise RuntimeError(
                f"S1 coverage failure: missing_matching={missing_m}, missing_candidate={missing_c}, "
                f"extra_matching={extra_m}, extra_candidate={extra_c}"
            )

        # Explode candidate IDs once. Empty lists are ignored.
        con.execute("""
            CREATE OR REPLACE TEMP TABLE candidate_ids AS
            SELECT DISTINCT c.source1_entity_id, trim(x.id) AS entity_id
            FROM c,
            UNNEST(string_split(COALESCE(c.candidate_entity_ids, ''), ',')) AS x(id)
            WHERE trim(x.id) <> ''
        """)

        # Explode match IDs once.
        con.execute("""
            CREATE OR REPLACE TEMP TABLE match_ids AS
            SELECT DISTINCT m.source1_entity_id, trim(x.id) AS entity_id
            FROM m,
            UNNEST(string_split(COALESCE(m.matched_entity_ids, ''), ',')) AS x(id)
            WHERE trim(x.id) <> ''
        """)

        bad_candidate_dupes = int(con.execute("""
            SELECT COUNT(*) FROM (
                SELECT source1_entity_id, entity_id, COUNT(*) AS n
                FROM (
                    SELECT c.source1_entity_id, trim(x.id) AS entity_id
                    FROM c,
                    UNNEST(string_split(COALESCE(c.candidate_entity_ids, ''), ',')) AS x(id)
                    WHERE trim(x.id) <> ''
                )
                GROUP BY 1,2 HAVING COUNT(*) > 1
            )
        """).fetchone()[0])
        bad_match_dupes = int(con.execute("""
            SELECT COUNT(*) FROM (
                SELECT source1_entity_id, entity_id, COUNT(*) AS n
                FROM (
                    SELECT m.source1_entity_id, trim(x.id) AS entity_id
                    FROM m,
                    UNNEST(string_split(COALESCE(m.matched_entity_ids, ''), ',')) AS x(id)
                    WHERE trim(x.id) <> ''
                )
                GROUP BY 1,2 HAVING COUNT(*) > 1
            )
        """).fetchone()[0])
        if bad_candidate_dupes or bad_match_dupes:
            raise RuntimeError(
                f"duplicate IDs inside rows: candidates={bad_candidate_dupes:,}, matches={bad_match_dupes:,}"
            )

        invalid_candidate_ids = int(con.execute("""
            SELECT COUNT(*)
            FROM candidate_ids
            WHERE NOT ((entity_id LIKE 'S2-%' AND entity_id IN (SELECT entity_id FROM s2))
                    OR (entity_id LIKE 'S3-%' AND entity_id IN (SELECT entity_id FROM s3)))
        """).fetchone()[0])
        invalid_match_ids = int(con.execute("""
            SELECT COUNT(*)
            FROM match_ids
            WHERE NOT ((entity_id LIKE 'S2-%' AND entity_id IN (SELECT entity_id FROM s2))
                    OR (entity_id LIKE 'S3-%' AND entity_id IN (SELECT entity_id FROM s3)))
        """).fetchone()[0])
        if invalid_candidate_ids or invalid_match_ids:
            raise RuntimeError(
                f"invalid/nonexistent IDs: candidates={invalid_candidate_ids:,}, matches={invalid_match_ids:,}"
            )

        outside_candidates = int(con.execute("""
            SELECT COUNT(*)
            FROM match_ids m
            WHERE NOT EXISTS (
                SELECT 1 FROM candidate_ids c
                WHERE c.source1_entity_id=m.source1_entity_id
                  AND c.entity_id=m.entity_id
            )
        """).fetchone()[0])
        if outside_candidates:
            raise RuntimeError(f"matches outside candidate set: {outside_candidates:,}")

        total_candidates = int(con.execute("SELECT COUNT(*) FROM candidate_ids").fetchone()[0])
        total_matches = int(con.execute("SELECT COUNT(*) FROM match_ids").fetchone()[0])
        empty_matches = int(con.execute("""
            SELECT COUNT(*) FROM m WHERE COALESCE(matched_entity_ids,'')=''
        """).fetchone()[0])
        empty_candidates = int(con.execute("""
            SELECT COUNT(*) FROM c WHERE COALESCE(candidate_entity_ids,'')=''
        """).fetchone()[0])

        print("=" * 88)
        print("SUBMISSION VALIDATION PASSED")
        print("=" * 88)
        print(f"S1 rows              : {s1_count:,}")
        print(f"matching_results rows: {int(m_count):,}")
        print(f"candidate_pairs rows : {int(c_count):,}")
        print(f"candidate IDs        : {total_candidates:,}")
        print(f"matched IDs          : {total_matches:,}")
        print(f"empty candidate rows : {empty_candidates:,}")
        print(f"empty match rows     : {empty_matches:,}")
        print("=" * 88)
    finally:
        con.close()


if __name__ == '__main__':
    main()

from __future__ import annotations

"""
Fast local submission validator for Amazon ML Challenge 2026.

Validates:
- exactly one row per TEST Source-1 entity
- exact S1 coverage
- S2/S3 ID syntax and existence
- no duplicate IDs inside a row
- final matches are a subset of candidate_pairs
"""

import argparse
from pathlib import Path


def load_tsv(path: Path, expected_header: tuple[str, str]):
    with path.open("r", encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split("\t")
        if tuple(header) != expected_header:
            raise RuntimeError(f"{path.name}: header must be {expected_header}, got {header}")

        rows = {}
        for line_no, line in enumerate(f, 2):
            line = line.rstrip("\r\n")
            parts = line.split("\t")
            if len(parts) != 2:
                raise RuntimeError(f"{path.name}:{line_no}: expected 2 TSV columns")
            key, value = parts
            if key in rows:
                raise RuntimeError(f"{path.name}:{line_no}: duplicate Source-1 ID {key}")
            rows[key] = value
    return rows


def split_ids(value: str):
    if value == "":
        return []
    out = [x for x in value.split(",") if x != ""]
    if len(out) != len(set(out)):
        raise RuntimeError("Duplicate matched/candidate ID inside one row")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--matching", default="artifacts/submission/matching_results.tsv")
    ap.add_argument("--candidate", default="artifacts/submission/candidate_pairs.tsv")
    ap.add_argument("--test-s1", default="artifacts/normalized/test_s1.parquet")
    ap.add_argument("--test-s2", default="artifacts/normalized/test_s2.parquet")
    ap.add_argument("--test-s3", default="artifacts/normalized/test_s3.parquet")
    args = ap.parse_args()

    import duckdb

    matching = Path(args.matching)
    candidate = Path(args.candidate)

    m = load_tsv(matching, ("source1_entity_id", "matched_entity_ids"))
    c = load_tsv(candidate, ("source1_entity_id", "candidate_entity_ids"))

    con = duckdb.connect()
    try:
        def ids(path):
            return set(
                r[0] for r in con.execute(
                    f"SELECT entity_id FROM read_parquet('{str(path.resolve()).replace(chr(92),'/').replace(chr(39),chr(39)+chr(39))}')"
                ).fetchall()
            )

        s1 = ids(Path(args.test_s1))
        s2 = ids(Path(args.test_s2))
        s3 = ids(Path(args.test_s3))
    finally:
        con.close()

    if set(m) != s1:
        raise RuntimeError(
            f"matching_results S1 mismatch: rows={len(m):,}, expected={len(s1):,}"
        )
    if set(c) != s1:
        raise RuntimeError(
            f"candidate_pairs S1 mismatch: rows={len(c):,}, expected={len(s1):,}"
        )

    candidate_sets = {}
    for sid, value in c.items():
        ids_ = split_ids(value)
        bad = [x for x in ids_ if not (x.startswith("S2-") or x.startswith("S3-"))]
        if bad:
            raise RuntimeError(f"{sid}: invalid candidate ID(s): {bad[:3]}")
        if any(x not in s2 and x not in s3 for x in ids_):
            raise RuntimeError(f"{sid}: candidate ID not present in test S2/S3")
        candidate_sets[sid] = set(ids_)

    total_matches = 0
    for sid, value in m.items():
        ids_ = split_ids(value)
        for x in ids_:
            if x not in s2 and x not in s3:
                raise RuntimeError(f"{sid}: match ID not present in test S2/S3: {x}")
            if x not in candidate_sets[sid]:
                raise RuntimeError(f"{sid}: match {x} is not in candidate set")
        total_matches += len(ids_)

    print("=" * 88)
    print("SUBMISSION VALIDATION PASSED")
    print("=" * 88)
    print(f"S1 rows              : {len(s1):,}")
    print(f"matching_results rows: {len(m):,}")
    print(f"candidate_pairs rows : {len(c):,}")
    print(f"matched IDs          : {total_matches:,}")
    print("=" * 88)


if __name__ == "__main__":
    main()

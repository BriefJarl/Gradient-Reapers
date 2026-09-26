from __future__ import annotations

"""
Production TEST feature builder
Amazon ML Challenge 2026 - Business Entity Resolution

Reuses the exact frozen feature contract from src.features.pair_features.
No pandas. DuckDB-native joins. Resumable per source.
"""

import argparse
import os
import sys
from pathlib import Path

import duckdb

# Direct script execution (python .\src\inference\...) does not put the
# repository root on sys.path. Add it explicitly so the import is robust.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from src.features.pair_features import build_feature_select, feature_column_names
except ModuleNotFoundError:
    from pair_features import build_feature_select, feature_column_names

NORMALIZED = ROOT / "artifacts" / "normalized"
BLOCKING = ROOT / "artifacts" / "blocking"
FEATURE_DIR = ROOT / "artifacts" / "features" / "final_test"
TMP_DIR = BLOCKING / "duckdb_tmp"

CANDIDATES = BLOCKING / "final_test" / "final_candidates.parquet"
S1 = NORMALIZED / "test_s1.parquet"
S2 = NORMALIZED / "test_s2.parquet"
S3 = NORMALIZED / "test_s3.parquet"

ROW_GROUP_SIZE = 250_000
THREADS = int(os.environ.get("DUCKDB_THREADS", "8"))
MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY", "6GB")


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def count_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{sql_quote(path)}')"
    ).fetchone()[0])


def get_columns(con: duckdb.DuckDBPyConnection, path: Path) -> set[str]:
    return {
        r[0] for r in con.execute(
            f"DESCRIBE SELECT * FROM read_parquet('{sql_quote(path)}')"
        ).fetchall()
    }


def configure(con: duckdb.DuckDBPyConnection) -> None:
    FEATURE_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{sql_quote(TMP_DIR)}'")


def output_for(source: str) -> Path:
    return FEATURE_DIR / f"test_features_{source.lower()}.parquet"


def partial_for(source: str) -> Path:
    return FEATURE_DIR / f"test_features_{source.lower()}.partial.parquet"


def validate_output(con, path: Path, source: str, expected_rows: int) -> None:
    cols = get_columns(con, path)
    expected = set(feature_column_names())
    missing = expected - cols
    if missing:
        raise RuntimeError(f"{source}: missing columns: {sorted(missing)}")

    rows = count_rows(con, path)
    if rows != expected_rows:
        raise RuntimeError(
            f"{source}: expected {expected_rows:,} rows, got {rows:,}"
        )

    checks = con.execute(f"""
        SELECT
            COUNT(*) FILTER (WHERE source1_entity_id IS NULL),
            COUNT(*) FILTER (WHERE matched_entity_id IS NULL),
            COUNT(*) FILTER (WHERE matched_source <> '{source}'),
            COUNT(*) FILTER (
                WHERE name_exact IS NULL
                   OR address_exact IS NULL
                   OR country_exact IS NULL
                   OR name_char_ratio IS NULL
                   OR address_char_ratio IS NULL
            ),
            COUNT(*) FILTER (
                WHERE name_char_ratio < 0 OR name_char_ratio > 1
                   OR address_char_ratio < 0 OR address_char_ratio > 1
                   OR name_token_overlap < 0 OR name_token_overlap > 1
                   OR name_token_jaccard < 0 OR name_token_jaccard > 1
                   OR address_token_overlap < 0 OR address_token_overlap > 1
                   OR address_token_jaccard < 0 OR address_token_jaccard > 1
                   OR name_length_ratio < 0 OR name_length_ratio > 1
                   OR address_length_ratio < 0 OR address_length_ratio > 1
            )
        FROM read_parquet('{sql_quote(path)}')
    """).fetchone()

    if any(int(x) != 0 for x in checks):
        raise RuntimeError(
            f"{source}: validation failed: "
            f"s1_null={checks[0]}, target_null={checks[1]}, "
            f"source={checks[2]}, feature_null={checks[3]}, "
            f"bad_range={checks[4]}"
        )


def build_source(con, source: str, force: bool) -> Path:
    out = output_for(source)
    partial = partial_for(source)

    if out.exists() and not force:
        try:
            candidate_count = int(con.execute(f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_quote(CANDIDATES)}')
                WHERE matched_source='{source}'
            """).fetchone()[0])
            validate_output(con, out, source, candidate_count)
            print(f"[{source}] Valid existing feature file -> SKIP")
            return out
        except Exception as exc:
            print(f"[{source}] Existing file invalid -> rebuild: {exc}")

    if partial.exists():
        partial.unlink()

    target = S2 if source == "S2" else S3
    expected_rows = int(con.execute(f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_quote(CANDIDATES)}')
        WHERE matched_source='{source}'
    """).fetchone()[0])

    s1_cols = get_columns(con, S1)
    target_cols = get_columns(con, target)

    select_features = build_feature_select(
        s1_alias="s1",
        target_alias="t",
        candidate_alias="c",
        s1_columns=s1_cols,
        target_columns=target_cols,
    )

    print("\n" + "=" * 88)
    print(f"BUILDING TEST FEATURES: {source}")
    print("=" * 88)
    print(f"Candidates : {expected_rows:,}")
    print(f"Output     : {out}")
    print(f"DuckDB     : {THREADS} threads / {MEMORY_LIMIT}")

    con.execute(f"""
        COPY (
            SELECT
                {select_features}
            FROM (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    num_blocking_methods,
                    blocking_methods
                FROM read_parquet('{sql_quote(CANDIDATES)}')
                WHERE matched_source='{source}'
            ) c
            INNER JOIN read_parquet('{sql_quote(S1)}') s1
                ON c.source1_entity_id=s1.entity_id
            INNER JOIN read_parquet('{sql_quote(target)}') t
                ON c.matched_entity_id=t.entity_id
        )
        TO '{sql_quote(partial)}'
        (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
    """)

    validate_output(con, partial, source, expected_rows)
    partial.replace(out)
    print(f"[{source}] COMPLETE: {expected_rows:,}")
    return out


def combine(con, s2_out: Path, s3_out: Path, force: bool) -> Path:
    out = FEATURE_DIR / "test_features.parquet"
    partial = FEATURE_DIR / "test_features.partial.parquet"
    expected = count_rows(con, CANDIDATES)

    if out.exists() and not force:
        if count_rows(con, out) == expected:
            print(f"[FINAL] Existing combined features valid: {expected:,}")
            return out

    if partial.exists():
        partial.unlink()

    con.execute(f"""
        COPY (
            SELECT * FROM read_parquet('{sql_quote(s2_out)}')
            UNION ALL
            SELECT * FROM read_parquet('{sql_quote(s3_out)}')
        )
        TO '{sql_quote(partial)}'
        (FORMAT PARQUET, COMPRESSION SNAPPY, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
    """)

    rows = count_rows(con, partial)
    if rows != expected:
        raise RuntimeError(f"Combined feature rows: expected {expected:,}, got {rows:,}")

    partial.replace(out)
    print(f"[FINAL] Combined TEST features: {rows:,}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for p in (CANDIDATES, S1, S2, S3):
        if not p.exists():
            raise FileNotFoundError(p)

    con = duckdb.connect()
    try:
        configure(con)
        candidate_rows = count_rows(con, CANDIDATES)
        print(f"TEST candidate rows: {candidate_rows:,}")

        s2 = build_source(con, "S2", args.force)
        s3 = build_source(con, "S3", args.force)
        final = combine(con, s2, s3, args.force)

        print("\n" + "=" * 88)
        print("FINAL TEST FEATURE BUILD PASSED")
        print("=" * 88)
        print(f"S2       : {s2}")
        print(f"S3       : {s3}")
        print(f"COMBINED : {final}")
        print(f"ROWS     : {count_rows(con, final):,}")
    finally:
        con.close()


if __name__ == "__main__":
    main()

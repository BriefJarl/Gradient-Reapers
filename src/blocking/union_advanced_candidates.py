from __future__ import annotations

"""
Amazon ML Challenge 2026: Advanced Candidate Union Pipeline.

Memory-safe, partitioned by target source (S2, then S3):
1. Unions candidates for S2:
   - Base optimized candidates
   - rare_name candidates
   - num_pfx candidates
   - domain_name candidates
2. Unions candidates for S3.
3. Streams S2 + S3 into {split}_expanded_candidates.parquet.

Zero cross-source Cartesian product, strictly under 8GB RAM.
"""

import argparse
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
CANDIDATE_DIR = BLOCKING_DIR / "candidates"
UNION_DIR = BLOCKING_DIR / "union"

UNION_DIR.mkdir(parents=True, exist_ok=True)

THREADS = 4
MEMORY_LIMIT = "8GB"
ROW_GROUP_SIZE = 250_000


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def get_con() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order = false")
    temp_dir = BLOCKING_DIR / "duckdb_tmp"
    temp_dir.mkdir(parents=True, exist_ok=True)
    con.execute(f"SET temp_directory = '{sql_quote(temp_dir)}'")
    return con


def union_source_candidates(
    con: duckdb.DuckDBPyConnection,
    split: str,
    target_source: str,
    out_name: str = "phase3_candidates",
) -> Path:
    """Union and deduplicate candidates for a single target source (S2 or S3)."""
    out_part = UNION_DIR / f"{split}_{out_name}_{target_source.lower()}.parquet"
    out_sql = sql_quote(out_part)

    base_path = UNION_DIR / f"{split}_optimized_candidates.parquet"
    rare_name_path = CANDIDATE_DIR / f"{split}_rare_name_candidates.parquet"
    num_pfx_path = CANDIDATE_DIR / f"{split}_num_pfx_candidates.parquet"
    core_name_path = CANDIDATE_DIR / f"{split}_core_name_candidates.parquet"
    domain_path = CANDIDATE_DIR / f"{split}_domain_name_candidates.parquet"
    indic_path = CANDIDATE_DIR / f"{split}_indic_translit_candidates.parquet"

    print(f"\n--- Unioning {split.upper()} -> {target_source} ({out_name}) ---")

    sources = []
    if base_path.exists():
        sources.append(f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                blocking_mask::INTEGER AS blocking_mask
            FROM read_parquet('{sql_quote(base_path)}')
            WHERE matched_source = '{target_source}'
        """)

    if rare_name_path.exists():
        sources.append(f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                8::INTEGER AS blocking_mask
            FROM read_parquet('{sql_quote(rare_name_path)}')
            WHERE matched_source = '{target_source}'
        """)

    if num_pfx_path.exists():
        sources.append(f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                32::INTEGER AS blocking_mask
            FROM read_parquet('{sql_quote(num_pfx_path)}')
            WHERE matched_source = '{target_source}'
        """)

    if core_name_path.exists():
        sources.append(f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                64::INTEGER AS blocking_mask
            FROM read_parquet('{sql_quote(core_name_path)}')
            WHERE matched_source = '{target_source}'
        """)

    if domain_path.exists():
        sources.append(f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                128::INTEGER AS blocking_mask
            FROM read_parquet('{sql_quote(domain_path)}')
            WHERE matched_source = '{target_source}'
        """)

    if indic_path.exists():
        sources.append(f"""
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                256::INTEGER AS blocking_mask
            FROM read_parquet('{sql_quote(indic_path)}')
            WHERE matched_source = '{target_source}'
        """)

    union_all_sql = "\nUNION ALL\n".join(sources)

    query = f"""
    COPY (
        WITH raw_union AS (
            {union_all_sql}
        )
        SELECT
            source1_entity_id,
            matched_entity_id,
            matched_source,
            bit_or(blocking_mask)::INTEGER AS blocking_mask,
            count(*)::TINYINT AS num_blocking_methods,
            CASE
                WHEN bit_or(blocking_mask) & 1 <> 0 THEN 'address'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 2 <> 0 THEN '+address_compact'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 4 <> 0 THEN '+name'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 8 <> 0 THEN '+rare_name'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 16 <> 0 THEN '+rare_address'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 32 <> 0 THEN '+num_pfx'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 64 <> 0 THEN '+core_name'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 128 <> 0 THEN '+domain_name'
                ELSE ''
            END ||
            CASE
                WHEN bit_or(blocking_mask) & 256 <> 0 THEN '+indic_translit'
                ELSE ''
            END AS blocking_methods
        FROM raw_union
        GROUP BY source1_entity_id, matched_entity_id, matched_source
    ) TO '{out_sql}' (
        FORMAT PARQUET,
        COMPRESSION SNAPPY,
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    );
    """
    con.execute(query)
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"{target_source} candidates: {count:,} -> {out_part}")
    return out_part


def combine_target_parts(
    con: duckdb.DuckDBPyConnection,
    split: str,
    s2_part: Path,
    s3_part: Path,
    out_name: str = "phase3_candidates",
) -> Path:
    """Concatenate S2 and S3 candidate parts into the final expanded candidate file."""
    final_out = UNION_DIR / f"{split}_{out_name}.parquet"
    final_sql = sql_quote(final_out)
    s2_sql = sql_quote(s2_part)
    s3_sql = sql_quote(s3_part)

    print(f"\nCombining S2 + S3 into final: {final_out}...")
    query = f"""
    COPY (
        SELECT * FROM read_parquet('{s2_sql}')
        UNION ALL
        SELECT * FROM read_parquet('{s3_sql}')
    ) TO '{final_sql}' (
        FORMAT PARQUET,
        COMPRESSION SNAPPY,
        ROW_GROUP_SIZE {ROW_GROUP_SIZE}
    );
    """
    con.execute(query)
    total = con.execute(f"SELECT COUNT(*) FROM read_parquet('{final_sql}')").fetchone()[0]
    print(f"Total Expanded Candidates ({split}): {total:,} -> {final_out}")
    return final_out


def run_split(con: duckdb.DuckDBPyConnection, split: str, out_name: str = "phase3_candidates") -> Path:
    print("\n" + "=" * 80)
    print(f"PHASE 3 CANDIDATE UNION: {split.upper()} ({out_name})")
    print("=" * 80)
    s2_part = union_source_candidates(con, split=split, target_source="S2", out_name=out_name)
    s3_part = union_source_candidates(con, split=split, target_source="S3", out_name=out_name)
    final_out = combine_target_parts(con, split=split, s2_part=s2_part, s3_part=s3_part, out_name=out_name)
    return final_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Union advanced candidate blocks safely.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both", help="Split to union.")
    parser.add_argument("--out-name", type=str, default="phase3_candidates", help="Base name of output parquet file.")
    args = parser.parse_args()

    con = get_con()
    try:
        splits = ["train", "test"] if args.split == "both" else [args.split]
        for sp in splits:
            run_split(con, split=sp, out_name=args.out_name)
    finally:
        con.close()

    print("\n" + "=" * 80)
    print("PHASE 3 CANDIDATE UNION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

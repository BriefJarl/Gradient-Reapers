from __future__ import annotations

import argparse
from pathlib import Path
import duckdb

ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
CANDIDATE_DIR = BLOCKING_DIR / "candidates"
INDEX_DIR = BLOCKING_DIR / "indexes"
UNION_DIR = BLOCKING_DIR / "union"

CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)
INDEX_DIR.mkdir(parents=True, exist_ok=True)
UNION_DIR.mkdir(parents=True, exist_ok=True)

THREADS = 8
MEMORY_LIMIT = "8GB"
RARE_TOKEN_TOP_K = 2
MAX_TOKEN_FREQ = 50

STRIP_SUFFIX_SQL = """
    trim(regexp_replace(
        strip_accents(name_norm),
        '\\b(private limited|pvt ltd|pvt limited|private ltd|ltd|limited|llc|inc|incorporated|corp|corporation|co|company|pllc|services|associates|group)\\b.*$',
        '',
        'g'
    ))
"""

CLEAN_DOMAIN_SQL = """
    regexp_replace(
        regexp_replace(strip_accents(name_compact), '^(www|http|https)', '', 'g'),
        '(com|org|net|in|co|io|biz|info)$', '', 'g'
    )
"""


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


def generate_num_pfx(con: duckdb.DuckDBPyConnection, split: str = "train") -> Path:
    """Street number + 3-char name prefix block."""
    out_path = CANDIDATE_DIR / f"{split}_num_pfx_candidates.parquet"
    out_sql = sql_quote(out_path)

    s1_path = NORMALIZED_DIR / f"{split}_s1.parquet"
    s2_path = NORMALIZED_DIR / f"{split}_s2.parquet"
    s3_path = NORMALIZED_DIR / f"{split}_s3.parquet"

    print("\n" + "=" * 80)
    print(f"GENERATING NUM_PFX CANDIDATES ({split.upper()})")
    print("=" * 80)

    query = f"""
    COPY (
        WITH s1 AS (
            SELECT entity_id, country_norm,
                   lower(regexp_extract(name_norm, '^([a-z0-9]{{3}})', 1)) AS name_pfx,
                   address_numeric_tokens[1] AS num1
            FROM read_parquet('{sql_quote(s1_path)}')
            WHERE len(address_numeric_tokens) > 0 AND country_norm <> ''
        ),
        targets AS (
            SELECT 'S2' AS src, entity_id, country_norm,
                   lower(regexp_extract(strip_accents(name_norm), '^([a-z0-9]{{3}})', 1)) AS name_pfx,
                   address_numeric_tokens[1] AS num1
            FROM read_parquet('{sql_quote(s2_path)}')
            WHERE len(address_numeric_tokens) > 0 AND country_norm <> ''
            UNION ALL
            SELECT 'S3' AS src, entity_id, country_norm,
                   lower(regexp_extract(strip_accents(name_norm), '^([a-z0-9]{{3}})', 1)) AS name_pfx,
                   address_numeric_tokens[1] AS num1
            FROM read_parquet('{sql_quote(s3_path)}')
            WHERE len(address_numeric_tokens) > 0 AND country_norm <> ''
        )
        SELECT DISTINCT
            s1.entity_id AS source1_entity_id,
            t.entity_id AS matched_entity_id,
            t.src AS matched_source,
            32 AS block_bit,
            'num_pfx' AS block_name
        FROM s1
        INNER JOIN targets t
            ON s1.country_norm = t.country_norm
           AND s1.name_pfx = t.name_pfx
           AND s1.num1 = t.num1
           AND s1.name_pfx <> ''
           AND s1.num1 <> ''
    ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"num_pfx candidates: {count:,} -> {out_path}")
    return out_path


def generate_core_name(con: duckdb.DuckDBPyConnection, split: str = "train") -> Path:
    """Exact match on core name with legal suffixes stripped."""
    out_path = CANDIDATE_DIR / f"{split}_core_name_candidates.parquet"
    out_sql = sql_quote(out_path)

    s1_path = NORMALIZED_DIR / f"{split}_s1.parquet"
    s2_path = NORMALIZED_DIR / f"{split}_s2.parquet"
    s3_path = NORMALIZED_DIR / f"{split}_s3.parquet"

    print("\n" + "=" * 80)
    print(f"GENERATING CORE_NAME CANDIDATES ({split.upper()})")
    print("=" * 80)

    query = f"""
    COPY (
        WITH s1 AS (
            SELECT entity_id, country_norm, {STRIP_SUFFIX_SQL} AS core_name
            FROM read_parquet('{sql_quote(s1_path)}')
            WHERE name_norm <> ''
        ),
        targets AS (
            SELECT 'S2' AS src, entity_id, country_norm, {STRIP_SUFFIX_SQL} AS core_name
            FROM read_parquet('{sql_quote(s2_path)}')
            WHERE name_norm <> ''
            UNION ALL
            SELECT 'S3' AS src, entity_id, country_norm, {STRIP_SUFFIX_SQL} AS core_name
            FROM read_parquet('{sql_quote(s3_path)}')
            WHERE name_norm <> ''
        )
        SELECT DISTINCT
            s1.entity_id AS source1_entity_id,
            t.entity_id AS matched_entity_id,
            t.src AS matched_source,
            64 AS block_bit,
            'core_name' AS block_name
        FROM s1
        INNER JOIN targets t
            ON s1.country_norm = t.country_norm
           AND s1.core_name = t.core_name
           AND LENGTH(s1.core_name) >= 4
    ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"core_name candidates: {count:,} -> {out_path}")
    return out_path


def generate_domain_name(con: duckdb.DuckDBPyConnection, split: str = "train") -> Path:
    """Exact match on compact name after stripping web extensions."""
    out_path = CANDIDATE_DIR / f"{split}_domain_name_candidates.parquet"
    out_sql = sql_quote(out_path)

    s1_path = NORMALIZED_DIR / f"{split}_s1.parquet"
    s2_path = NORMALIZED_DIR / f"{split}_s2.parquet"
    s3_path = NORMALIZED_DIR / f"{split}_s3.parquet"

    print("\n" + "=" * 80)
    print(f"GENERATING DOMAIN_NAME CANDIDATES ({split.upper()})")
    print("=" * 80)

    query = f"""
    COPY (
        WITH s1 AS (
            SELECT entity_id, country_norm, {CLEAN_DOMAIN_SQL} AS name_domain
            FROM read_parquet('{sql_quote(s1_path)}')
            WHERE name_compact <> ''
        ),
        targets AS (
            SELECT 'S2' AS src, entity_id, country_norm, {CLEAN_DOMAIN_SQL} AS name_domain
            FROM read_parquet('{sql_quote(s2_path)}')
            WHERE name_compact <> ''
            UNION ALL
            SELECT 'S3' AS src, entity_id, country_norm, {CLEAN_DOMAIN_SQL} AS name_domain
            FROM read_parquet('{sql_quote(s3_path)}')
            WHERE name_compact <> ''
        )
        SELECT DISTINCT
            s1.entity_id AS source1_entity_id,
            t.entity_id AS matched_entity_id,
            t.src AS matched_source,
            128 AS block_bit,
            'domain_name' AS block_name
        FROM s1
        INNER JOIN targets t
            ON s1.country_norm = t.country_norm
           AND s1.name_domain = t.name_domain
           AND LENGTH(s1.name_domain) >= 5
    ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"domain_name candidates: {count:,} -> {out_path}")
    return out_path


def build_rare_name_indexes_if_missing(con: duckdb.DuckDBPyConnection, split: str = "test") -> dict[str, Path]:
    """Ensure rare name token indexes exist for S2 and S3."""
    targets = {
        "S2": NORMALIZED_DIR / f"{split}_s2.parquet",
        "S3": NORMALIZED_DIR / f"{split}_s3.parquet",
    }
    paths = {}
    for src, p in targets.items():
        idx_path = INDEX_DIR / f"{split}_{src.lower()}_rare_name_token_index.parquet"
        if not idx_path.exists():
            print(f"Building rare name token index for {src} ({split})...")
            q = f"""
            COPY (
                WITH exploded AS (
                    SELECT DISTINCT
                        CAST(entity_id AS VARCHAR) AS entity_id,
                        country_norm,
                        LOWER(TRIM(token)) AS token
                    FROM read_parquet('{sql_quote(p)}')
                    CROSS JOIN UNNEST(name_tokens) AS u(token)
                    WHERE country_norm <> '' AND token <> ''
                      AND NOT regexp_matches(token, '^[0-9]+$')
                ),
                freq AS (
                    SELECT country_norm, token, COUNT(*) AS token_freq
                    FROM exploded
                    GROUP BY country_norm, token
                    HAVING COUNT(*) <= {MAX_TOKEN_FREQ}
                )
                SELECT e.entity_id, e.country_norm, e.token, f.token_freq
                FROM exploded e
                INNER JOIN freq f ON e.country_norm = f.country_norm AND e.token = f.token
            ) TO '{sql_quote(idx_path)}' (FORMAT PARQUET, COMPRESSION ZSTD);
            """
            con.execute(q)
        paths[src] = idx_path
    return paths


def generate_rare_name_test(con: duckdb.DuckDBPyConnection) -> Path:
    """Generate rare_name candidates for test split."""
    build_rare_name_indexes_if_missing(con, split="test")
    out_path = CANDIDATE_DIR / "test_rare_name_candidates.parquet"
    out_sql = sql_quote(out_path)

    s1_path = NORMALIZED_DIR / "test_s1.parquet"
    s2_idx = INDEX_DIR / "test_s2_rare_name_token_index.parquet"
    s3_idx = INDEX_DIR / "test_s3_rare_name_token_index.parquet"

    print("\n" + "=" * 80)
    print("GENERATING RARE_NAME CANDIDATES (TEST)")
    print("=" * 80)

    query = f"""
    COPY (
        WITH s1_tokens AS (
            SELECT DISTINCT
                entity_id AS source1_entity_id,
                country_norm,
                LOWER(TRIM(token)) AS token
            FROM read_parquet('{sql_quote(s1_path)}')
            CROSS JOIN UNNEST(name_tokens) AS u(token)
            WHERE country_norm <> '' AND token <> ''
        ),
        s2_match AS (
            SELECT DISTINCT s.source1_entity_id, s.country_norm, s.token, i.token_freq
            FROM s1_tokens s
            INNER JOIN read_parquet('{sql_quote(s2_idx)}') i
                ON s.country_norm = i.country_norm AND s.token = i.token
        ),
        s2_top AS (
            SELECT source1_entity_id, country_norm, token
            FROM s2_match
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY source1_entity_id
                ORDER BY token_freq ASC, LENGTH(token) DESC, token ASC
            ) <= {RARE_TOKEN_TOP_K}
        ),
        s3_match AS (
            SELECT DISTINCT s.source1_entity_id, s.country_norm, s.token, i.token_freq
            FROM s1_tokens s
            INNER JOIN read_parquet('{sql_quote(s3_idx)}') i
                ON s.country_norm = i.country_norm AND s.token = i.token
        ),
        s3_top AS (
            SELECT source1_entity_id, country_norm, token
            FROM s3_match
            QUALIFY ROW_NUMBER() OVER (
                PARTITION BY source1_entity_id
                ORDER BY token_freq ASC, LENGTH(token) DESC, token ASC
            ) <= {RARE_TOKEN_TOP_K}
        )
        SELECT DISTINCT
            s.source1_entity_id,
            i.entity_id AS matched_entity_id,
            'S2' AS matched_source,
            8 AS block_bit,
            'rare_name' AS block_name
        FROM s2_top s
        INNER JOIN read_parquet('{sql_quote(s2_idx)}') i
            ON s.country_norm = i.country_norm AND s.token = i.token
        UNION ALL
        SELECT DISTINCT
            s.source1_entity_id,
            i.entity_id AS matched_entity_id,
            'S3' AS matched_source,
            8 AS block_bit,
            'rare_name' AS block_name
        FROM s3_top s
        INNER JOIN read_parquet('{sql_quote(s3_idx)}') i
            ON s.country_norm = i.country_norm AND s.token = i.token
    ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """
    con.execute(query)
    count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
    print(f"rare_name test candidates: {count:,} -> {out_path}")
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate advanced candidate pairs.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="train", help="Data split to generate for.")
    args = parser.parse_args()

    con = get_con()
    try:
        splits = ["train", "test"] if args.split == "both" else [args.split]
        for sp in splits:
            print("\n" + "#" * 80)
            print(f"PROCESSING SPLIT: {sp.upper()}")
            print("#" * 80)
            generate_num_pfx(con, split=sp)
            generate_core_name(con, split=sp)
            generate_domain_name(con, split=sp)
            if sp == "test":
                generate_rare_name_test(con)
    finally:
        con.close()

    print("\n" + "=" * 80)
    print("ADVANCED CANDIDATE GENERATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

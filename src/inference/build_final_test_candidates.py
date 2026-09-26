from __future__ import annotations

"""
Production test candidate builder
Amazon ML Challenge 2026 - Business Entity Resolution

Goal
----
Build the final TEST candidate pool using the same measured candidate-generation
families that produced the final TRAIN pool:

1. Exact address
2. Exact compact address
3. Exact name
4. Exact compact name
5. Rare-address candidates, with the TRAIN-validated B_MEDIUM deterministic filter
6. ASCII/transliteration expansion
7. Strict rare-name expansion

Important
---------
- No pandas.
- DuckDB-first.
- Parquet intermediates.
- Disk spilling enabled.
- Deterministic.
- Resumable where practical.
- Never uses test labels / ground truth.
- Does NOT force top-1.
- Preserves candidate provenance.
- Final pair identity:
    (source1_entity_id, matched_entity_id, matched_source)

The ASCII and strict-rare-name expansion files are produced by the already
validated src/blocking/expand_recall.py implementation. If they are absent,
this script runs that implementation automatically.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

NORMALIZED = ROOT / "artifacts" / "normalized"
BLOCKING = ROOT / "artifacts" / "blocking"
FINAL_DIR = BLOCKING / "final_test"
EXPERIMENTS = BLOCKING / "experiments"
TMP_DIR = BLOCKING / "duckdb_tmp"

S1 = NORMALIZED / "test_s1.parquet"
S2 = NORMALIZED / "test_s2.parquet"
S3 = NORMALIZED / "test_s3.parquet"

ASCII_EXP = EXPERIMENTS / "test_ascii_exact_candidates.parquet"
RARE_NAME_EXP = EXPERIMENTS / "test_rare_name_strict_candidates.parquet"

BASE_RAW = FINAL_DIR / "base_raw.parquet"
RARE_ADDRESS_RAW = FINAL_DIR / "rare_address_raw.parquet"
BASE_OPT = FINAL_DIR / "base_optimized.parquet"
EXP_UNION = FINAL_DIR / "expansion_union.parquet"
FINAL_CANDIDATES = FINAL_DIR / "final_candidates.parquet"
STATS_JSON = FINAL_DIR / "candidate_stats.json"

EXPECTED_S1 = 1_732_544
EXPECTED_S2 = 4_887_273
EXPECTED_S3 = 5_082_316

# Keep the same operating envelope used successfully during the large train
# pipeline. Environment variables allow controlled tuning without editing code.
# The machine has ~9-10 GB RAM available during the competition run.
# The previous 12-thread / 8 GB setting was sufficient for joins, but the
# provenance GROUP BY in build_optimized_base created too many concurrent
# hash-table partitions and exhausted RAM.  Use a conservative default that
# leaves headroom for Windows + DuckDB spill buffers.
THREADS = int(os.environ.get("DUCKDB_THREADS", "8"))
MEMORY_LIMIT = os.environ.get("DUCKDB_MEMORY", "6GB")
ROW_GROUP_SIZE = 250_000

# Provenance bit masks.
BIT_ADDRESS = 1
BIT_ADDRESS_COMPACT = 2
BIT_NAME = 4
BIT_RARE_NAME = 8
BIT_RARE_ADDRESS = 16
BIT_ASCII = 32
BIT_NUMERIC = 64
BIT_STRICT_RARE_NAME = 128

STOP_TOKENS = {
    "llc",
    "ltd",
    "limited",
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "co",
    "company",
    "pvt",
    "private",
    "plc",
    "llp",
    "lp",
    "the",
    "and",
}


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def sql_str(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def configure(con: duckdb.DuckDBPyConnection) -> None:
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    TMP_DIR.mkdir(parents=True, exist_ok=True)

    con.execute(f"SET threads={THREADS}")
    con.execute(f"SET memory_limit='{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order=false")
    con.execute("SET enable_progress_bar=true")
    con.execute(f"SET temp_directory='{sql_path(TMP_DIR)}'")


def require_inputs() -> None:
    missing = [p for p in (S1, S2, S3) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing normalized TEST parquet(s):\n  - "
            + "\n  - ".join(str(p) for p in missing)
        )


def count_rows(con: duckdb.DuckDBPyConnection, path: Path) -> int:
    return int(
        con.execute(
            f"SELECT COUNT(*) FROM read_parquet('{sql_path(path)}')"
        ).fetchone()[0]
    )


def ensure_test_expansions(force: bool) -> None:
    """
    Reuse the already validated recall-expansion implementation.

    We intentionally do not duplicate the transliteration implementation here.
    That avoids creating two subtly different definitions of ASCII blocking.
    """
    ascii_ok = ASCII_EXP.exists() and not force
    rare_ok = RARE_NAME_EXP.exists() and not force

    if ascii_ok and rare_ok:
        print("\n[EXPANSION] Existing TEST ASCII + strict rare-name files found.")
        print("[EXPANSION] Reusing them.")
        return

    script = ROOT / "src" / "blocking" / "expand_recall.py"
    if not script.exists():
        raise FileNotFoundError(
            f"Validated expansion script not found: {script}"
        )

    print("\n" + "=" * 88)
    print("BUILDING TEST RECALL EXPANSIONS")
    print("=" * 88)
    print("Methods : ascii,rare_name")
    print("Script  :", script)

    cmd = [
        sys.executable,
        str(script),
        "--dataset",
        "test",
        "--methods",
        "ascii,rare_name",
    ]

    subprocess.run(cmd, cwd=str(ROOT), check=True)

    if not ASCII_EXP.exists():
        raise RuntimeError(f"ASCII expansion was not created: {ASCII_EXP}")
    if not RARE_NAME_EXP.exists():
        raise RuntimeError(
            f"Strict rare-name expansion was not created: {RARE_NAME_EXP}"
        )


def build_exact_base(
    con: duckdb.DuckDBPyConnection,
    force: bool,
) -> None:
    if BASE_RAW.exists() and not force:
        print(f"\n[BASE] Reusing {BASE_RAW}")
        print(f"[BASE] Rows: {count_rows(con, BASE_RAW):,}")
        return

    if BASE_RAW.exists():
        BASE_RAW.unlink()

    p1, p2, p3 = map(sql_path, (S1, S2, S3))

    print("\n" + "=" * 88)
    print("BUILDING EXACT TEST BLOCKING")
    print("=" * 88)

    # We deliberately generate each blocking family separately and UNION ALL
    # them. Provenance is aggregated later. This avoids one giant OR join and
    # makes each relational operation easy for DuckDB to optimize.
    query = f"""
    WITH
    s1 AS (
        SELECT
            entity_id,
            country_norm,
            name_norm,
            name_compact,
            address_norm,
            address_compact
        FROM read_parquet('{p1}')
    ),
    s2 AS (
        SELECT
            entity_id,
            country_norm,
            name_norm,
            name_compact,
            address_norm,
            address_compact
        FROM read_parquet('{p2}')
    ),
    s3 AS (
        SELECT
            entity_id,
            country_norm,
            name_norm,
            name_compact,
            address_norm,
            address_compact
        FROM read_parquet('{p3}')
    ),

    address_s2 AS (
        SELECT
            a.entity_id AS source1_entity_id,
            b.entity_id AS matched_entity_id,
            'S2' AS matched_source,
            {BIT_ADDRESS}::INTEGER AS blocking_mask,
            'address' AS blocking_methods
        FROM s1 a
        INNER JOIN s2 b
          ON a.country_norm = b.country_norm
         AND a.address_norm IS NOT NULL
         AND b.address_norm IS NOT NULL
         AND a.address_norm <> ''
         AND b.address_norm <> ''
         AND a.address_norm = b.address_norm
    ),

    address_s3 AS (
        SELECT
            a.entity_id,
            b.entity_id,
            'S3',
            {BIT_ADDRESS},
            'address'
        FROM s1 a
        INNER JOIN s3 b
          ON a.country_norm = b.country_norm
         AND a.address_norm IS NOT NULL
         AND b.address_norm IS NOT NULL
         AND a.address_norm <> ''
         AND b.address_norm <> ''
         AND a.address_norm = b.address_norm
    ),

    compact_address_s2 AS (
        SELECT
            a.entity_id,
            b.entity_id,
            'S2',
            {BIT_ADDRESS_COMPACT},
            'address_compact'
        FROM s1 a
        INNER JOIN s2 b
          ON a.country_norm = b.country_norm
         AND a.address_compact IS NOT NULL
         AND b.address_compact IS NOT NULL
         AND a.address_compact <> ''
         AND b.address_compact <> ''
         AND a.address_compact = b.address_compact
    ),

    compact_address_s3 AS (
        SELECT
            a.entity_id,
            b.entity_id,
            'S3',
            {BIT_ADDRESS_COMPACT},
            'address_compact'
        FROM s1 a
        INNER JOIN s3 b
          ON a.country_norm = b.country_norm
         AND a.address_compact IS NOT NULL
         AND b.address_compact IS NOT NULL
         AND a.address_compact <> ''
         AND b.address_compact <> ''
         AND a.address_compact = b.address_compact
    ),

    name_s2 AS (
        SELECT
            a.entity_id,
            b.entity_id,
            'S2',
            {BIT_NAME},
            'name'
        FROM s1 a
        INNER JOIN s2 b
          ON a.country_norm = b.country_norm
         AND a.name_norm IS NOT NULL
         AND b.name_norm IS NOT NULL
         AND a.name_norm <> ''
         AND b.name_norm <> ''
         AND a.name_norm = b.name_norm
    ),

    name_s3 AS (
        SELECT
            a.entity_id,
            b.entity_id,
            'S3',
            {BIT_NAME},
            'name'
        FROM s1 a
        INNER JOIN s3 b
          ON a.country_norm = b.country_norm
         AND a.name_norm IS NOT NULL
         AND b.name_norm IS NOT NULL
         AND a.name_norm <> ''
         AND b.name_norm <> ''
         AND a.name_norm = b.name_norm
    ),

    compact_name_s2 AS (
        SELECT
            a.entity_id,
            b.entity_id,
            'S2',
            {BIT_NAME},
            'name_compact'
        FROM s1 a
        INNER JOIN s2 b
          ON a.country_norm = b.country_norm
         AND a.name_compact IS NOT NULL
         AND b.name_compact IS NOT NULL
         AND a.name_compact <> ''
         AND b.name_compact <> ''
         AND a.name_compact = b.name_compact
    ),

    compact_name_s3 AS (
        SELECT
            a.entity_id,
            b.entity_id,
            'S3',
            {BIT_NAME},
            'name_compact'
        FROM s1 a
        INNER JOIN s3 b
          ON a.country_norm = b.country_norm
         AND a.name_compact IS NOT NULL
         AND b.name_compact IS NOT NULL
         AND a.name_compact <> ''
         AND b.name_compact <> ''
         AND a.name_compact = b.name_compact
    )

    SELECT * FROM address_s2
    UNION ALL SELECT * FROM address_s3
    UNION ALL SELECT * FROM compact_address_s2
    UNION ALL SELECT * FROM compact_address_s3
    UNION ALL SELECT * FROM name_s2
    UNION ALL SELECT * FROM name_s3
    UNION ALL SELECT * FROM compact_name_s2
    UNION ALL SELECT * FROM compact_name_s3
    """

    partial = BASE_RAW.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()

    con.execute(
        f"""
        COPY ({query})
        TO '{sql_path(partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
        """
    )

    rows = count_rows(con, partial)
    print(f"[BASE] Raw exact-block rows: {rows:,}")

    partial.replace(BASE_RAW)


def build_rare_address(
    con: duckdb.DuckDBPyConnection,
    force: bool,
) -> None:
    """
    Reproduce the TRAIN candidate strategy's rare-address family.

    The broad token join can be large. We do NOT retain all broad candidates.
    Instead we immediately keep only pairs satisfying the validated B_MEDIUM
    policy:
        address token overlap >= 3
        OR exact/compact name agreement.

    Token frequency is country-aware and capped at 50, matching the project's
    established rare-token design.
    """
    if RARE_ADDRESS_RAW.exists() and not force:
        print(f"\n[RARE_ADDRESS] Reusing {RARE_ADDRESS_RAW}")
        print(
            f"[RARE_ADDRESS] Rows: "
            f"{count_rows(con, RARE_ADDRESS_RAW):,}"
        )
        return

    if RARE_ADDRESS_RAW.exists():
        RARE_ADDRESS_RAW.unlink()

    p1, p2, p3 = map(sql_path, (S1, S2, S3))
    stop_sql = ", ".join(sql_str(x) for x in sorted(STOP_TOKENS))

    print("\n" + "=" * 88)
    print("BUILDING TEST RARE-ADDRESS BLOCKING")
    print("=" * 88)
    print("Frequency cap : <= 50 per country")
    print("Keep rule     : overlap >= 3 OR exact/compact name")

    # Raw target token tables are deduplicated per entity/token first.
    # This prevents repeated tokens from inflating overlap.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s2_addr_tokens AS
        WITH raw AS (
            SELECT DISTINCT
                entity_id,
                country_norm,
                LOWER(TRIM(u.token)) AS token
            FROM read_parquet('{p2}'),
                 UNNEST(address_tokens) AS u(token)
            WHERE token IS NOT NULL
              AND LENGTH(TRIM(token)) >= 4
              AND LOWER(TRIM(token)) NOT IN ({stop_sql})
        ),
        freq AS (
            SELECT
                country_norm,
                token,
                COUNT(*) AS freq
            FROM raw
            GROUP BY 1,2
            HAVING COUNT(*) <= 50
        )
        SELECT r.entity_id, r.country_norm, r.token
        FROM raw r
        INNER JOIN freq f
          ON r.country_norm = f.country_norm
         AND r.token = f.token
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s3_addr_tokens AS
        WITH raw AS (
            SELECT DISTINCT
                entity_id,
                country_norm,
                LOWER(TRIM(u.token)) AS token
            FROM read_parquet('{p3}'),
                 UNNEST(address_tokens) AS u(token)
            WHERE token IS NOT NULL
              AND LENGTH(TRIM(token)) >= 4
              AND LOWER(TRIM(token)) NOT IN ({stop_sql})
        ),
        freq AS (
            SELECT
                country_norm,
                token,
                COUNT(*) AS freq
            FROM raw
            GROUP BY 1,2
            HAVING COUNT(*) <= 50
        )
        SELECT r.entity_id, r.country_norm, r.token
        FROM raw r
        INNER JOIN freq f
          ON r.country_norm = f.country_norm
         AND r.token = f.token
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP VIEW s1_addr_tokens AS
        SELECT DISTINCT
            entity_id,
            country_norm,
            LOWER(TRIM(u.token)) AS token
        FROM read_parquet('{p1}'),
             UNNEST(address_tokens) AS u(token)
        WHERE token IS NOT NULL
          AND LENGTH(TRIM(token)) >= 4
          AND LOWER(TRIM(token)) NOT IN ({stop_sql})
        """
    )

    # The pair aggregation is the expensive part. It is deliberately staged
    # and spilled to DuckDB temp storage instead of being materialized in Python.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE rare_address_pairs AS
        SELECT
            a.entity_id AS source1_entity_id,
            b.entity_id AS matched_entity_id,
            'S2' AS matched_source,
            COUNT(*)::INTEGER AS address_overlap
        FROM s1_addr_tokens a
        INNER JOIN s2_addr_tokens b
          ON a.country_norm = b.country_norm
         AND a.token = b.token
        GROUP BY 1,2

        UNION ALL

        SELECT
            a.entity_id,
            b.entity_id,
            'S3',
            COUNT(*)::INTEGER
        FROM s1_addr_tokens a
        INNER JOIN s3_addr_tokens b
          ON a.country_norm = b.country_norm
         AND a.token = b.token
        GROUP BY 1,2
        """
    )

    print(
        "[RARE_ADDRESS] Broad token-overlap pair rows: "
        f"{con.execute('SELECT COUNT(*) FROM rare_address_pairs').fetchone()[0]:,}"
    )

    # Enrich only the rare-address pair relation. The final B_MEDIUM filter is
    # intentionally source-independent and therefore valid for TEST.
    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE rare_address_filtered AS
        SELECT
            r.source1_entity_id,
            r.matched_entity_id,
            r.matched_source,
            {BIT_RARE_ADDRESS}::INTEGER AS blocking_mask,
            1::TINYINT AS num_blocking_methods,
            'rare_address' AS blocking_methods
        FROM rare_address_pairs r
        INNER JOIN read_parquet('{p1}') a
          ON r.source1_entity_id = a.entity_id
        INNER JOIN (
            SELECT
                entity_id,
                name_norm,
                name_compact
            FROM read_parquet('{p2}')

            UNION ALL

            SELECT
                entity_id,
                name_norm,
                name_compact
            FROM read_parquet('{p3}')
        ) b
          ON r.matched_entity_id = b.entity_id
        WHERE
            r.address_overlap >= 3
            OR (
                a.name_norm IS NOT NULL
                AND b.name_norm IS NOT NULL
                AND a.name_norm = b.name_norm
            )
            OR (
                a.name_compact IS NOT NULL
                AND b.name_compact IS NOT NULL
                AND a.name_compact = b.name_compact
            )
        """
    )

    partial = RARE_ADDRESS_RAW.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                blocking_mask,
                num_blocking_methods,
                blocking_methods
            FROM rare_address_filtered
        )
        TO '{sql_path(partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
        """
    )

    rows = count_rows(con, partial)
    print(f"[RARE_ADDRESS] Filtered rows: {rows:,}")

    partial.replace(RARE_ADDRESS_RAW)


def build_optimized_base(
    con: duckdb.DuckDBPyConnection,
    force: bool,
) -> None:
    """
    Memory-safe implementation of the TRAIN-validated B_MEDIUM policy.

    The previous implementation joined ~22M exact candidates back to all
    normalized S1/S2/S3 rows and then GROUP BY'd the result.  That is
    unnecessary: the exact blocking rows already contain the provenance needed
    for the B_MEDIUM decision.

    We therefore:
      1. aggregate exact-block provenance only;
      2. keep multi-method exact pairs;
      3. keep exact-name / compact-name-only pairs;
      4. append the already-filtered rare-address relation;
      5. do the final pair-level provenance merge later.

    This removes the largest memory hotspot from the failed run.
    """
    EXACT_OPT = FINAL_DIR / "exact_optimized.parquet"
    RARE_OPT = FINAL_DIR / "rare_address_optimized.parquet"

    if BASE_OPT.exists() and not force:
        print(f"\n[BASE OPT] Reusing {BASE_OPT}")
        print(f"[BASE OPT] Rows: {count_rows(con, BASE_OPT):,}")
        return

    for pth in (EXACT_OPT, RARE_OPT, BASE_OPT):
        if pth.exists():
            pth.unlink()

    p = sql_path(BASE_RAW)
    r = sql_path(RARE_ADDRESS_RAW)

    print("\n" + "=" * 88)
    print("APPLYING TRAIN-VALIDATED B_MEDIUM POLICY TO TEST BASE")
    print("=" * 88)
    print("Memory-safe mode: no normalized-table rejoin")

    # ------------------------------------------------------------------
    # 1) Exact-family optimization.
    #
    # The raw exact relation contains:
    #   address
    #   address_compact
    #   name
    #   name_compact
    #
    # B_MEDIUM:
    #   - retain every candidate supported by >=2 blocking methods
    #   - retain name/name_compact-only candidates
    #   - discard address-only candidates
    #
    # We aggregate only the 22M-row candidate relation.  No 5M-row target
    # table is joined into this operation.
    # ------------------------------------------------------------------
    exact_partial = EXACT_OPT.with_suffix(".partial.parquet")
    if exact_partial.exists():
        exact_partial.unlink()

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                BIT_OR(blocking_mask)::INTEGER AS blocking_mask,
                COUNT(DISTINCT blocking_methods)::TINYINT
                    AS num_blocking_methods,
                STRING_AGG(
                    DISTINCT blocking_methods,
                    '|'
                    ORDER BY blocking_methods
                ) AS blocking_methods
            FROM read_parquet('{p}')
            GROUP BY
                source1_entity_id,
                matched_entity_id,
                matched_source
            HAVING
                COUNT(DISTINCT blocking_methods) >= 2
                OR
                BOOL_OR(
                    blocking_methods IN ('name', 'name_compact')
                )
        )
        TO '{sql_path(exact_partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
        """
    )

    exact_rows = count_rows(con, exact_partial)
    exact_partial.replace(EXACT_OPT)

    print(f"[BASE OPT] Exact optimized rows : {exact_rows:,}")

    # ------------------------------------------------------------------
    # 2) Rare-address relation.
    #
    # rare_address_raw has ALREADY been filtered using the validated:
    #     overlap >= 3 OR exact/compact name
    #
    # There is no reason to rejoin the normalized tables here.
    # ------------------------------------------------------------------
    rare_partial = RARE_OPT.with_suffix(".partial.parquet")
    if rare_partial.exists():
        rare_partial.unlink()

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                BIT_OR(blocking_mask)::INTEGER AS blocking_mask,
                1::TINYINT AS num_blocking_methods,
                'rare_address' AS blocking_methods
            FROM read_parquet('{r}')
            GROUP BY
                source1_entity_id,
                matched_entity_id,
                matched_source
        )
        TO '{sql_path(rare_partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
        """
    )

    rare_rows = count_rows(con, rare_partial)
    rare_partial.replace(RARE_OPT)

    print(f"[BASE OPT] Rare-address rows   : {rare_rows:,}")

    # ------------------------------------------------------------------
    # 3) Pair-level deduplication across exact + rare-address.
    #
    # This relation is still much smaller than the previous
    # normalized-table join.  Provenance is merged here so downstream
    # features see the correct block mask.
    # ------------------------------------------------------------------
    base_partial = BASE_OPT.with_suffix(".partial.parquet")
    if base_partial.exists():
        base_partial.unlink()

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                BIT_OR(blocking_mask)::INTEGER AS blocking_mask,
                COUNT(DISTINCT blocking_methods)::TINYINT
                    AS num_blocking_methods,
                STRING_AGG(
                    DISTINCT blocking_methods,
                    '|'
                    ORDER BY blocking_methods
                ) AS blocking_methods
            FROM (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    blocking_methods
                FROM read_parquet('{sql_path(EXACT_OPT)}')

                UNION ALL

                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    blocking_methods
                FROM read_parquet('{sql_path(RARE_OPT)}')
            ) x
            GROUP BY
                source1_entity_id,
                matched_entity_id,
                matched_source
        )
        TO '{sql_path(base_partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
        """
    )

    base_rows = count_rows(con, base_partial)
    base_partial.replace(BASE_OPT)

    print(f"[BASE OPT] Final optimized base: {base_rows:,}")

    # Hard uniqueness check at this stage.  This is cheap relative to feature
    # generation and prevents duplicate candidate identities propagating.
    duplicate_groups = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source
                FROM read_parquet('{sql_path(BASE_OPT)}')
                GROUP BY 1,2,3
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )

    if duplicate_groups:
        raise RuntimeError(
            f"[BASE OPT] Duplicate candidate groups detected: "
            f"{duplicate_groups:,}"
        )

    print("[BASE OPT] Pair uniqueness check: PASSED")

def build_expansion_union(
    con: duckdb.DuckDBPyConnection,
    force: bool,
) -> None:
    if EXP_UNION.exists() and not force:
        print(f"\n[EXPANSION UNION] Reusing {EXP_UNION}")
        print(
            f"[EXPANSION UNION] Rows: "
            f"{count_rows(con, EXP_UNION):,}"
        )
        return

    if EXP_UNION.exists():
        EXP_UNION.unlink()

    a = sql_path(ASCII_EXP)
    r = sql_path(RARE_NAME_EXP)

    print("\n" + "=" * 88)
    print("BUILDING TEST EXPANSION UNION")
    print("=" * 88)

    partial = EXP_UNION.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()

    # Each expansion file has already been generated as unique pair rows.
    # We still deduplicate here because defensive pair-level uniqueness is
    # cheap relative to the downstream feature generation cost.
    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                BIT_OR(blocking_mask)::INTEGER AS blocking_mask,
                COUNT(*)::TINYINT AS num_blocking_methods,
                STRING_AGG(
                    DISTINCT blocking_methods,
                    '|'
                    ORDER BY blocking_methods
                ) AS blocking_methods
            FROM (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    blocking_methods
                FROM read_parquet('{a}')

                UNION ALL

                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    blocking_methods
                FROM read_parquet('{r}')
            ) x
            GROUP BY 1,2,3
        )
        TO '{sql_path(partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
        """
    )

    rows = count_rows(con, partial)
    print(f"[EXPANSION UNION] Unique expansion pairs: {rows:,}")

    partial.replace(EXP_UNION)


def _blocking_method_count_sql(expr: str) -> str:
    """
    Count pipe-delimited provenance methods without a Python row loop.
    Upstream relations are already pair-unique and their provenance is merged.
    """
    return f"CASE WHEN {expr} IS NULL OR {expr} = '' THEN 0 ELSE ARRAY_LENGTH(STRING_SPLIT({expr}, '|')) END"


def build_final(
    con: duckdb.DuckDBPyConnection,
    force: bool,
) -> None:
    """
    Assemble the TEST candidate pool with a resume-first strategy.

    Important optimization
    ----------------------
    BASE_OPT and EXP_UNION are already pair-unique and provenance-aggregated.
    EXP_UNION is anti-joined against BASE_OPT before it becomes part of the
    final pool. Therefore a second 42M-row GROUP BY / STRING_AGG provenance
    merge is redundant.

    The previous implementation spent ~20 minutes performing that unnecessary
    final GROUP BY and then failed in the statistics query because the outer
    SELECT referenced matched_source after projecting it away.

    This implementation:
      1. Reuses final_premerge.parquet if the previous run created it.
      2. Adds num_blocking_methods directly from blocking_methods.
      3. Otherwise builds the final pool with one anti-join + UNION ALL.
      4. Computes statistics with a valid two-pass aggregation.
    """
    if FINAL_CANDIDATES.exists() and not force:
        print(f"\n[FINAL] Reusing {FINAL_CANDIDATES}")
        print(f"[FINAL] Rows: {count_rows(con, FINAL_CANDIDATES):,}")
        return

    if FINAL_CANDIDATES.exists():
        FINAL_CANDIDATES.unlink()

    b = sql_path(BASE_OPT)
    e = sql_path(EXP_UNION)
    raw_final = FINAL_DIR / "final_premerge.parquet"

    print("\n" + "=" * 88)
    print("BUILDING FINAL TEST CANDIDATE POOL")
    print("=" * 88)

    # ------------------------------------------------------------------
    # Fast resume path.
    #
    # The failed run already completed final_premerge.parquet before the
    # BinderException in the statistics query. Reuse it instead of paying
    # another 15-25 minutes for the same anti-join.
    # ------------------------------------------------------------------
    if raw_final.exists() and not force:
        print(f"[FINAL] Reusing existing premerge: {raw_final}")
        print("[FINAL] Skipping expensive base/expansion rebuild.")
        source_relation = sql_path(raw_final)
    else:
        raw_partial = raw_final.with_suffix(".partial.parquet")
        if raw_final.exists():
            raw_final.unlink()
        if raw_partial.exists():
            raw_partial.unlink()

        print("[FINAL] Building base + NEW expansion candidates...")
        con.execute(
            f"""
            COPY (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source,
                    blocking_mask,
                    num_blocking_methods,
                    blocking_methods
                FROM read_parquet('{b}')

                UNION ALL

                SELECT
                    e.source1_entity_id,
                    e.matched_entity_id,
                    e.matched_source,
                    e.blocking_mask,
                    e.num_blocking_methods,
                    e.blocking_methods
                FROM read_parquet('{e}') e
                ANTI JOIN read_parquet('{b}') base
                  ON e.source1_entity_id = base.source1_entity_id
                 AND e.matched_entity_id = base.matched_entity_id
                 AND e.matched_source = base.matched_source
            )
            TO '{sql_path(raw_partial)}'
            (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
            """
        )
        raw_partial.replace(raw_final)
        source_relation = sql_path(raw_final)

    # ------------------------------------------------------------------
    # Materialize the canonical six-column candidate contract.
    #
    # The previous final_premerge file has five columns because the old code
    # intentionally added num_blocking_methods only during the redundant
    # provenance merge. Derive it deterministically from blocking_methods.
    # ------------------------------------------------------------------
    partial = FINAL_CANDIDATES.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()

    method_count = _blocking_method_count_sql("blocking_methods")

    con.execute(
        f"""
        COPY (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                blocking_mask::INTEGER AS blocking_mask,
                COALESCE(
                    num_blocking_methods::TINYINT,
                    {method_count}::TINYINT
                ) AS num_blocking_methods,
                blocking_methods
            FROM read_parquet('{source_relation}')
        )
        TO '{sql_path(partial)}'
        (FORMAT PARQUET, COMPRESSION ZSTD, ROW_GROUP_SIZE {ROW_GROUP_SIZE})
        """
    )

    rows = count_rows(con, partial)
    partial.replace(FINAL_CANDIDATES)

    print(f"[FINAL] Candidate rows: {rows:,}")

    # ------------------------------------------------------------------
    # HARD VALIDATION
    #
    # BASE_OPT and EXP_UNION were independently pair-unique; expansion rows
    # are anti-joined against BASE_OPT. Thus the final construction itself
    # guarantees pair uniqueness. We still run a direct duplicate check as a
    # safety gate before feature generation.
    # ------------------------------------------------------------------
    dup_groups = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT
                    source1_entity_id,
                    matched_entity_id,
                    matched_source
                FROM read_parquet('{sql_path(FINAL_CANDIDATES)}')
                GROUP BY 1,2,3
                HAVING COUNT(*) > 1
            )
            """
        ).fetchone()[0]
    )

    bad_source = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_path(FINAL_CANDIDATES)}')
            WHERE matched_source NOT IN ('S2','S3')
            """
        ).fetchone()[0]
    )

    null_ids = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_path(FINAL_CANDIDATES)}')
            WHERE source1_entity_id IS NULL
               OR matched_entity_id IS NULL
               OR matched_source IS NULL
            """
        ).fetchone()[0]
    )

    bad_s1 = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM (
                SELECT DISTINCT source1_entity_id
                FROM read_parquet('{sql_path(FINAL_CANDIDATES)}')
            ) c
            ANTI JOIN read_parquet('{sql_path(S1)}') s
              ON c.source1_entity_id = s.entity_id
            """
        ).fetchone()[0]
    )

    bad_s2 = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_path(FINAL_CANDIDATES)}') c
            ANTI JOIN read_parquet('{sql_path(S2)}') s
              ON c.matched_entity_id = s.entity_id
            WHERE c.matched_source = 'S2'
            """
        ).fetchone()[0]
    )

    bad_s3 = int(
        con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_path(FINAL_CANDIDATES)}') c
            ANTI JOIN read_parquet('{sql_path(S3)}') s
              ON c.matched_entity_id = s.entity_id
            WHERE c.matched_source = 'S3'
            """
        ).fetchone()[0]
    )

    bad_target_ids = bad_s2 + bad_s3

    if dup_groups or bad_source or null_ids or bad_s1 or bad_target_ids:
        raise RuntimeError(
            "FINAL CANDIDATE VALIDATION FAILED: "
            f"duplicate_groups={dup_groups:,}, "
            f"bad_source={bad_source:,}, "
            f"null_ids={null_ids:,}, "
            f"bad_s1={bad_s1:,}, "
            f"bad_target_ids={bad_target_ids:,}"
        )

    # ------------------------------------------------------------------
    # Candidate distribution.
    #
    # FIX: matched_source must be read from the candidate relation. The old
    # query projected only source1_entity_id/cnt in its subquery and then
    # referenced matched_source outside that subquery, causing:
    # Binder Error: Referenced column "matched_source" not found.
    #
    # We deliberately separate the source totals from the per-S1 quantiles.
    # This is cheaper and, importantly, logically correct.
    # ------------------------------------------------------------------
    totals = con.execute(
        f"""
        SELECT
            COUNT(*)::BIGINT AS total_candidates,
            COUNT(DISTINCT source1_entity_id)::BIGINT AS covered_s1,
            COUNT(*) FILTER (WHERE matched_source='S2')::BIGINT AS s2_candidates,
            COUNT(*) FILTER (WHERE matched_source='S3')::BIGINT AS s3_candidates
        FROM read_parquet('{sql_path(FINAL_CANDIDATES)}')
        """
    ).fetchone()

    quantiles = con.execute(
        f"""
        SELECT
            AVG(cnt)::DOUBLE AS mean_per_s1,
            QUANTILE_CONT(cnt, 0.50)::DOUBLE AS p50,
            QUANTILE_CONT(cnt, 0.90)::DOUBLE AS p90,
            QUANTILE_CONT(cnt, 0.95)::DOUBLE AS p95,
            QUANTILE_CONT(cnt, 0.99)::DOUBLE AS p99,
            MAX(cnt)::BIGINT AS max_per_s1
        FROM (
            SELECT source1_entity_id, COUNT(*)::BIGINT AS cnt
            FROM read_parquet('{sql_path(FINAL_CANDIDATES)}')
            GROUP BY 1
        )
        """
    ).fetchone()

    total, covered, s2c, s3c = totals
    mean, p50, p90, p95, p99, maxc = quantiles

    stats = {
        "test_s1_rows": EXPECTED_S1,
        "test_s2_rows": EXPECTED_S2,
        "test_s3_rows": EXPECTED_S3,
        "final_candidate_rows": int(total),
        "covered_s1": int(covered),
        "s1_without_candidates": EXPECTED_S1 - int(covered),
        "s2_candidate_rows": int(s2c),
        "s3_candidate_rows": int(s3c),
        "mean_candidates_per_s1": float(mean or 0.0),
        "p50_candidates_per_s1": float(p50 or 0.0),
        "p90_candidates_per_s1": float(p90 or 0.0),
        "p95_candidates_per_s1": float(p95 or 0.0),
        "p99_candidates_per_s1": float(p99 or 0.0),
        "max_candidates_per_s1": int(maxc or 0),
        "duplicate_pair_groups": int(dup_groups),
        "invalid_source_rows": int(bad_source),
        "null_id_rows": int(null_ids),
        "invalid_s1_rows": int(bad_s1),
        "invalid_target_rows": int(bad_target_ids),
        "threads": THREADS,
        "memory_limit": MEMORY_LIMIT,
    }

    STATS_JSON.write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    print("\n" + "=" * 88)
    print("FINAL TEST CANDIDATE STATISTICS")
    print("=" * 88)
    print(f"S1 test entities       : {EXPECTED_S1:,}")
    print(f"S1 with candidates     : {int(covered):,}")
    print(f"S1 without candidates  : {EXPECTED_S1 - int(covered):,}")
    print(f"Total candidates       : {int(total):,}")
    print(f"S2 candidates          : {int(s2c):,}")
    print(f"S3 candidates          : {int(s3c):,}")
    print(f"Mean / S1              : {float(mean):,.3f}")
    print(f"P50 / S1               : {float(p50):,.3f}")
    print(f"P90 / S1               : {float(p90):,.3f}")
    print(f"P95 / S1               : {float(p95):,.3f}")
    print(f"P99 / S1               : {float(p99):,.3f}")
    print(f"MAX / S1               : {int(maxc):,}")
    print(f"Output                 : {FINAL_CANDIDATES}")
    print(f"Stats                  : {STATS_JSON}")

    print("\n" + "=" * 88)
    print("FINAL TEST CANDIDATE BUILD PASSED")
    print("=" * 88)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build final Amazon ML Challenge TEST candidate pool."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild all candidate intermediates.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    started = time.perf_counter()

    print("=" * 88)
    print("AMAZON ML CHALLENGE 2026")
    print("FINAL TEST CANDIDATE GENERATION")
    print("=" * 88)
    print(f"Threads      : {THREADS}")
    print(f"Memory       : {MEMORY_LIMIT}")
    print(f"Temp dir     : {TMP_DIR}")
    print(f"S1           : {S1}")
    print(f"S2           : {S2}")
    print(f"S3           : {S3}")

    require_inputs()

    con = duckdb.connect()
    try:
        configure(con)

        actual_s1 = count_rows(con, S1)
        actual_s2 = count_rows(con, S2)
        actual_s3 = count_rows(con, S3)

        if (actual_s1, actual_s2, actual_s3) != (
            EXPECTED_S1,
            EXPECTED_S2,
            EXPECTED_S3,
        ):
            raise RuntimeError(
                "TEST normalized row-count mismatch: "
                f"S1={actual_s1:,}, S2={actual_s2:,}, S3={actual_s3:,}"
            )

        ensure_test_expansions(args.force)
        build_exact_base(con, args.force)
        build_rare_address(con, args.force)
        build_optimized_base(con, args.force)
        build_expansion_union(con, args.force)
        build_final(con, args.force)

        elapsed = time.perf_counter() - started
        print(f"\nElapsed time: {elapsed/60:.2f} minutes")

    finally:
        con.close()


if __name__ == "__main__":
    main()

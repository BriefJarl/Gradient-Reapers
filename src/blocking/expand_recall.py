from __future__ import annotations

import argparse
from pathlib import Path

import duckdb
from unidecode import unidecode
from duckdb.sqltypes import VARCHAR


# ============================================================
# AMAZON ML CHALLENGE 2026
# RECALL EXPANSION EXPERIMENTS
#
# Methods:
#   1. ASCII / transliteration exact blocking
#   2. Numeric-anchor blocking
#   3. Strict rare-name blocking
#
# IMPORTANT:
# This script does NOT modify the existing 31.22M candidate pool.
# It only creates independent experiment candidate Parquets.
# ============================================================


ROOT = Path(__file__).resolve().parents[2]

NORMALIZED = ROOT / "artifacts" / "normalized"
BLOCKING = ROOT / "artifacts" / "blocking"
EXPERIMENTS = BLOCKING / "experiments"
TMP = BLOCKING / "duckdb_tmp"


TRAIN = {
    "S1": NORMALIZED / "train_s1.parquet",
    "S2": NORMALIZED / "train_s2.parquet",
    "S3": NORMALIZED / "train_s3.parquet",
}

TEST = {
    "S1": NORMALIZED / "test_s1.parquet",
    "S2": NORMALIZED / "test_s2.parquet",
    "S3": NORMALIZED / "test_s3.parquet",
}


def sql_path(path: Path) -> str:
    return str(path.resolve()).replace("\\", "/").replace("'", "''")


def configure(con: duckdb.DuckDBPyConnection) -> None:
    con.execute("SET threads=8")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET preserve_insertion_order=false")

    TMP.mkdir(parents=True, exist_ok=True)

    con.execute(
        f"SET temp_directory='{sql_path(TMP)}'"
    )

    con.execute("SET enable_progress_bar=true")


def register_unidecode(con: duckdb.DuckDBPyConnection) -> None:
    """
    Register Python transliteration function.

    The function is only called for strings containing non-ASCII
    characters. Existing ASCII names are reused directly.
    """

    def _unidecode(value):
        if value is None:
            return None
        value = str(value)
        if not value.strip():
            return None
        return unidecode(value)

    con.create_function(
        "py_unidecode",
        _unidecode,
        [VARCHAR],
        VARCHAR,
        null_handling="special",
    )


def target_path(dataset: str, source: str) -> Path:
    if dataset == "train":
        return TRAIN[source]
    return TEST[source]


def check_inputs(paths: dict[str, Path]) -> None:
    for source, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(
                f"{source} normalized parquet not found:\n{path}"
            )


# ============================================================
# ASCII / TRANSLITERATION
# ============================================================

def write_ascii(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    s1: Path,
    s2: Path,
    s3: Path,
) -> Path:

    out = EXPERIMENTS / f"{dataset}_ascii_exact_candidates.parquet"

    if out.exists():
        out.unlink()

    print("\n" + "=" * 80)
    print("ASCII / TRANSLITERATION BLOCKING")
    print("=" * 80)

    p1 = sql_path(s1)
    p2 = sql_path(s2)
    p3 = sql_path(s3)

    # --------------------------------------------------------
    # We intentionally derive ASCII from the EXISTING
    # name_norm field.
    #
    # We do NOT require name_ascii to exist.
    #
    # ASCII strings are reused directly.
    # Only non-ASCII strings call py_unidecode().
    # --------------------------------------------------------

    print("[ASCII] Preparing S1 name keys...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_ascii AS
        SELECT
            entity_id,
            country_norm,

            CASE
                WHEN name_norm IS NULL
                     OR TRIM(name_norm) = ''
                THEN NULL

                WHEN regexp_matches(
                    name_norm,
                    '[^\\\\x00-\\\\x7F]'
                )
                THEN LOWER(py_unidecode(name_norm))

                ELSE LOWER(name_norm)
            END AS name_ascii

        FROM read_parquet('{p1}')
        WHERE name_norm IS NOT NULL
          AND TRIM(name_norm) <> ''
        """
    )

    print("[ASCII] Preparing S2 name keys...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s2_ascii AS
        SELECT
            entity_id,
            country_norm,

            CASE
                WHEN name_norm IS NULL
                     OR TRIM(name_norm) = ''
                THEN NULL

                WHEN regexp_matches(
                    name_norm,
                    '[^\\\\x00-\\\\x7F]'
                )
                THEN LOWER(py_unidecode(name_norm))

                ELSE LOWER(name_norm)
            END AS name_ascii

        FROM read_parquet('{p2}')
        WHERE name_norm IS NOT NULL
          AND TRIM(name_norm) <> ''
        """
    )

    print("[ASCII] Preparing S3 name keys...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s3_ascii AS
        SELECT
            entity_id,
            country_norm,

            CASE
                WHEN name_norm IS NULL
                     OR TRIM(name_norm) = ''
                THEN NULL

                WHEN regexp_matches(
                    name_norm,
                    '[^\\\\x00-\\\\x7F]'
                )
                THEN LOWER(py_unidecode(name_norm))

                ELSE LOWER(name_norm)
            END AS name_ascii

        FROM read_parquet('{p3}')
        WHERE name_norm IS NOT NULL
          AND TRIM(name_norm) <> ''
        """
    )

    print("[ASCII] Building compact transliteration keys...")

    con.execute(
        """
        ALTER TABLE s1_ascii
        ADD COLUMN name_ascii_compact VARCHAR
        """
    )

    con.execute(
        """
        ALTER TABLE s2_ascii
        ADD COLUMN name_ascii_compact VARCHAR
        """
    )

    con.execute(
        """
        ALTER TABLE s3_ascii
        ADD COLUMN name_ascii_compact VARCHAR
        """
    )

    for table in ("s1_ascii", "s2_ascii", "s3_ascii"):
        con.execute(
            f"""
            UPDATE {table}
            SET name_ascii_compact =
                NULLIF(
                    regexp_replace(
                        name_ascii,
                        '[^a-z0-9]+',
                        '',
                        'g'
                    ),
                    ''
                )
            """
        )

    # --------------------------------------------------------
    # EXACT ASCII
    # + COMPACT ASCII
    #
    # Country remains part of the blocking key.
    # We are NOT relaxing country because profiling showed
    # country is complete in the available normalized data.
    # --------------------------------------------------------

    print("[ASCII] Joining S1 -> S2/S3...")

    query = """
    SELECT DISTINCT
        source1_entity_id,
        matched_entity_id,
        matched_source,
        CAST(32 AS INTEGER) AS blocking_mask,
        CAST(1 AS TINYINT) AS num_blocking_methods,
        'ascii_name' AS blocking_methods

    FROM
    (
        SELECT
            a.entity_id AS source1_entity_id,
            b.entity_id AS matched_entity_id,
            'S2' AS matched_source

        FROM s1_ascii a
        INNER JOIN s2_ascii b
          ON a.country_norm = b.country_norm
         AND a.name_ascii = b.name_ascii

        WHERE a.name_ascii IS NOT NULL
          AND b.name_ascii IS NOT NULL

        UNION ALL

        SELECT
            a.entity_id,
            b.entity_id,
            'S2'

        FROM s1_ascii a
        INNER JOIN s2_ascii b
          ON a.country_norm = b.country_norm
         AND a.name_ascii_compact = b.name_ascii_compact

        WHERE a.name_ascii_compact IS NOT NULL
          AND b.name_ascii_compact IS NOT NULL

        UNION ALL

        SELECT
            a.entity_id,
            b.entity_id,
            'S3'

        FROM s1_ascii a
        INNER JOIN s3_ascii b
          ON a.country_norm = b.country_norm
         AND a.name_ascii = b.name_ascii

        WHERE a.name_ascii IS NOT NULL
          AND b.name_ascii IS NOT NULL

        UNION ALL

        SELECT
            a.entity_id,
            b.entity_id,
            'S3'

        FROM s1_ascii a
        INNER JOIN s3_ascii b
          ON a.country_norm = b.country_norm
         AND a.name_ascii_compact = b.name_ascii_compact

        WHERE a.name_ascii_compact IS NOT NULL
          AND b.name_ascii_compact IS NOT NULL
    ) x
    """

    print(f"[ASCII] Writing: {out}")

    con.execute(
        f"""
        COPY ({query})
        TO '{sql_path(out)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_path(out)}')
        """
    ).fetchone()[0]

    print(f"[ASCII] Candidate rows: {count:,}")
    print(f"[ASCII] Saved: {out}")

    return out


# ============================================================
# NUMERIC ANCHOR BLOCKING
# ============================================================

def write_numeric(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    s1: Path,
    s2: Path,
    s3: Path,
) -> Path:

    out = EXPERIMENTS / f"{dataset}_numeric_anchor_candidates.parquet"

    if out.exists():
        out.unlink()

    print("\n" + "=" * 80)
    print("SAFE NUMERIC ANCHOR BLOCKING")
    print("=" * 80)

    p1 = sql_path(s1)
    p2 = sql_path(s2)
    p3 = sql_path(s3)

    # --------------------------------------------------------
    # DESIGN
    #
    # The previous numeric implementation used:
    #
    #   address numbers >= 3 digits
    #   name numbers >= 4 digits
    #   target frequency <= 2000
    #
    # That can create enormous Cartesian joins.
    #
    # This version is intentionally conservative:
    #
    #   address numeric token >= 4 digits
    #   name numeric token >= 5 digits
    #   target frequency <= 100
    #
    # Country remains part of the blocking key.
    #
    # This is an EXPERIMENT, not the final matcher.
    # --------------------------------------------------------

    MAX_TARGET_FREQ = 100

    print("[NUMERIC] Extracting S1 anchors...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_numeric AS

        SELECT DISTINCT
            entity_id,
            country_norm,
            anchor

        FROM
        (
            SELECT
                entity_id,
                country_norm,
                u.anchor

            FROM read_parquet('{p1}'),

            UNNEST(
                regexp_extract_all(
                    COALESCE(address_norm, ''),
                    '[0-9]{{4,}}'
                )
            ) AS u(anchor)

            UNION ALL

            SELECT
                entity_id,
                country_norm,
                u.anchor

            FROM read_parquet('{p1}'),

            UNNEST(
                regexp_extract_all(
                    COALESCE(name_norm, ''),
                    '[0-9]{{5,}}'
                )
            ) AS u(anchor)
        )

        WHERE anchor IS NOT NULL
          AND TRIM(anchor) <> ''
        """
    )

    print("[NUMERIC] Extracting S2 anchors...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s2_numeric_raw AS

        SELECT DISTINCT
            entity_id,
            country_norm,
            anchor

        FROM
        (
            SELECT
                entity_id,
                country_norm,
                u.anchor

            FROM read_parquet('{p2}'),

            UNNEST(
                regexp_extract_all(
                    COALESCE(address_norm, ''),
                    '[0-9]{{4,}}'
                )
            ) AS u(anchor)

            UNION ALL

            SELECT
                entity_id,
                country_norm,
                u.anchor

            FROM read_parquet('{p2}'),

            UNNEST(
                regexp_extract_all(
                    COALESCE(name_norm, ''),
                    '[0-9]{{5,}}'
                )
            ) AS u(anchor)
        )

        WHERE anchor IS NOT NULL
          AND TRIM(anchor) <> ''
        """
    )

    print("[NUMERIC] Extracting S3 anchors...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s3_numeric_raw AS

        SELECT DISTINCT
            entity_id,
            country_norm,
            anchor

        FROM
        (
            SELECT
                entity_id,
                country_norm,
                u.anchor

            FROM read_parquet('{p3}'),

            UNNEST(
                regexp_extract_all(
                    COALESCE(address_norm, ''),
                    '[0-9]{{4,}}'
                )
            ) AS u(anchor)

            UNION ALL

            SELECT
                entity_id,
                country_norm,
                u.anchor

            FROM read_parquet('{p3}'),

            UNNEST(
                regexp_extract_all(
                    COALESCE(name_norm, ''),
                    '[0-9]{{5,}}'
                )
            ) AS u(anchor)
        )

        WHERE anchor IS NOT NULL
          AND TRIM(anchor) <> ''
        """
    )

    # --------------------------------------------------------
    # FREQUENCY CONTROL
    #
    # Only target anchors appearing <= 100 times are allowed.
    # This prevents common numbers from generating massive joins.
    # --------------------------------------------------------

    print(
        "[NUMERIC] Applying strict target frequency cap..."
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s2_numeric AS

        SELECT r.*

        FROM s2_numeric_raw r

        INNER JOIN
        (
            SELECT
                country_norm,
                anchor

            FROM s2_numeric_raw

            GROUP BY
                country_norm,
                anchor

            HAVING COUNT(*) <= {MAX_TARGET_FREQ}

        ) f

          ON r.country_norm = f.country_norm
         AND r.anchor = f.anchor
        """
    )

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s3_numeric AS

        SELECT r.*

        FROM s3_numeric_raw r

        INNER JOIN
        (
            SELECT
                country_norm,
                anchor

            FROM s3_numeric_raw

            GROUP BY
                country_norm,
                anchor

            HAVING COUNT(*) <= {MAX_TARGET_FREQ}

        ) f

          ON r.country_norm = f.country_norm
         AND r.anchor = f.anchor
        """
    )

    # --------------------------------------------------------
    # IMPORTANT OPTIMIZATION
    #
    # Only retain S1 anchors that actually exist in S2/S3.
    # This avoids carrying irrelevant S1 numeric keys.
    # --------------------------------------------------------

    print(
        "[NUMERIC] Restricting S1 anchors to observed target anchors..."
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE s1_numeric_s2 AS

        SELECT DISTINCT
            a.entity_id,
            a.country_norm,
            a.anchor

        FROM s1_numeric a

        SEMI JOIN s2_numeric b

          ON a.country_norm = b.country_norm
         AND a.anchor = b.anchor
        """
    )

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE s1_numeric_s3 AS

        SELECT DISTINCT
            a.entity_id,
            a.country_norm,
            a.anchor

        FROM s1_numeric a

        SEMI JOIN s3_numeric b

          ON a.country_norm = b.country_norm
         AND a.anchor = b.anchor
        """
    )

    # --------------------------------------------------------
    # Write S2 and S3 independently.
    #
    # This avoids constructing one giant UNION + DISTINCT
    # intermediate relation.
    # --------------------------------------------------------

    print("[NUMERIC] Generating S2 numeric candidates...")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE numeric_s2_candidates AS

        SELECT DISTINCT
            a.entity_id AS source1_entity_id,
            b.entity_id AS matched_entity_id,
            'S2' AS matched_source

        FROM s1_numeric_s2 a

        INNER JOIN s2_numeric b

          ON a.country_norm = b.country_norm
         AND a.anchor = b.anchor
        """
    )

    s2_count = con.execute(
        """
        SELECT COUNT(*)
        FROM numeric_s2_candidates
        """
    ).fetchone()[0]

    print(
        f"[NUMERIC] S2 candidates: {s2_count:,}"
    )

    print("[NUMERIC] Generating S3 numeric candidates...")

    con.execute(
        """
        CREATE OR REPLACE TEMP TABLE numeric_s3_candidates AS

        SELECT DISTINCT
            a.entity_id AS source1_entity_id,
            b.entity_id AS matched_entity_id,
            'S3' AS matched_source

        FROM s1_numeric_s3 a

        INNER JOIN s3_numeric b

          ON a.country_norm = b.country_norm
         AND a.anchor = b.anchor
        """
    )

    s3_count = con.execute(
        """
        SELECT COUNT(*)
        FROM numeric_s3_candidates
        """
    ).fetchone()[0]

    print(
        f"[NUMERIC] S3 candidates: {s3_count:,}"
    )

    # --------------------------------------------------------
    # Final write.
    #
    # Since S2 and S3 are already separately deduplicated,
    # this UNION ALL does not require a huge global DISTINCT.
    # --------------------------------------------------------

    print("[NUMERIC] Writing final numeric experiment...")

    con.execute(
        f"""
        COPY
        (
            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                CAST(64 AS INTEGER) AS blocking_mask,
                CAST(1 AS TINYINT) AS num_blocking_methods,
                'numeric_anchor' AS blocking_methods

            FROM numeric_s2_candidates

            UNION ALL

            SELECT
                source1_entity_id,
                matched_entity_id,
                matched_source,
                CAST(64 AS INTEGER) AS blocking_mask,
                CAST(1 AS TINYINT) AS num_blocking_methods,
                'numeric_anchor' AS blocking_methods

            FROM numeric_s3_candidates
        )

        TO '{sql_path(out)}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        )
        """
    )

    total = s2_count + s3_count

    print()
    print(
        f"[NUMERIC] Candidate rows: {total:,}"
    )

    print(
        f"[NUMERIC] Saved: {out}"
    )

    return out


# ============================================================
# STRICT RARE NAME BLOCKING
# ============================================================

def write_rare_name(
    con: duckdb.DuckDBPyConnection,
    dataset: str,
    s1: Path,
    s2: Path,
    s3: Path,
) -> Path:

    out = EXPERIMENTS / f"{dataset}_rare_name_strict_candidates.parquet"

    if out.exists():
        out.unlink()

    print("\n" + "=" * 80)
    print("STRICT RARE-NAME BLOCKING")
    print("=" * 80)

    p1 = sql_path(s1)
    p2 = sql_path(s2)
    p3 = sql_path(s3)

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # We previously measured broad rare-name blocking:
    #
    # 20.8M candidates
    # only ~636K NEW true pairs
    # ~3.2% incremental purity
    #
    # Therefore this experiment is intentionally stricter.
    #
    # Token:
    #   length >= 5
    #
    # Frequency:
    #   <= 20 entities within country
    #
    # Corporate suffixes are excluded because they carry little
    # entity-specific information.
    # --------------------------------------------------------

    stop_tokens = (
        "'llc','ltd','limited','inc','incorporated',"
        "'corp','corporation','co','company','pvt','private',"
        "'plc','llp','lp','the','and'"
    )

    print("[RARE_NAME] Building S2 rare-token index...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s2_rare_tokens AS

        WITH raw AS
        (
            SELECT DISTINCT
                entity_id,
                country_norm,
                LOWER(TRIM(u.token)) AS token

            FROM read_parquet('{p2}'),

            UNNEST(name_tokens) AS u(token)

            WHERE token IS NOT NULL
              AND LENGTH(TRIM(token)) >= 5
              AND LOWER(TRIM(token))
                    NOT IN ({stop_tokens})
        ),

        rare AS
        (
            SELECT
                country_norm,
                token

            FROM raw

            GROUP BY
                country_norm,
                token

            HAVING COUNT(*) <= 20
        )

        SELECT
            r.entity_id,
            r.country_norm,
            r.token

        FROM raw r

        INNER JOIN rare f

          ON r.country_norm = f.country_norm
         AND r.token = f.token
        """
    )

    print("[RARE_NAME] Building S3 rare-token index...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s3_rare_tokens AS

        WITH raw AS
        (
            SELECT DISTINCT
                entity_id,
                country_norm,
                LOWER(TRIM(u.token)) AS token

            FROM read_parquet('{p3}'),

            UNNEST(name_tokens) AS u(token)

            WHERE token IS NOT NULL
              AND LENGTH(TRIM(token)) >= 5
              AND LOWER(TRIM(token))
                    NOT IN ({stop_tokens})
        ),

        rare AS
        (
            SELECT
                country_norm,
                token

            FROM raw

            GROUP BY
                country_norm,
                token

            HAVING COUNT(*) <= 20
        )

        SELECT
            r.entity_id,
            r.country_norm,
            r.token

        FROM raw r

        INNER JOIN rare f

          ON r.country_norm = f.country_norm
         AND r.token = f.token
        """
    )

    print("[RARE_NAME] Building S1 token relation...")

    con.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE s1_rare_tokens AS

        SELECT DISTINCT
            entity_id,
            country_norm,
            LOWER(TRIM(u.token)) AS token

        FROM read_parquet('{p1}'),

        UNNEST(name_tokens) AS u(token)

        WHERE token IS NOT NULL
          AND LENGTH(TRIM(token)) >= 5
          AND LOWER(TRIM(token))
                NOT IN ({stop_tokens})
        """
    )

    print("[RARE_NAME] Joining rare tokens...")

    query = """
    SELECT DISTINCT
        source1_entity_id,
        matched_entity_id,
        matched_source,
        CAST(128 AS INTEGER) AS blocking_mask,
        CAST(1 AS TINYINT) AS num_blocking_methods,
        'rare_name_strict' AS blocking_methods

    FROM
    (
        SELECT
            a.entity_id AS source1_entity_id,
            b.entity_id AS matched_entity_id,
            'S2' AS matched_source

        FROM s1_rare_tokens a

        INNER JOIN s2_rare_tokens b
          ON a.country_norm = b.country_norm
         AND a.token = b.token

        UNION ALL

        SELECT
            a.entity_id,
            b.entity_id,
            'S3'

        FROM s1_rare_tokens a

        INNER JOIN s3_rare_tokens b
          ON a.country_norm = b.country_norm
         AND a.token = b.token
    ) x
    """

    print(f"[RARE_NAME] Writing: {out}")

    con.execute(
        f"""
        COPY ({query})
        TO '{sql_path(out)}'
        (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet('{sql_path(out)}')
        """
    ).fetchone()[0]

    print(f"[RARE_NAME] Candidate rows: {count:,}")
    print(f"[RARE_NAME] Saved: {out}")

    return out


# ============================================================
# MAIN
# ============================================================

def main() -> None:

    parser = argparse.ArgumentParser(
        description="Amazon ML Challenge recall expansion experiments"
    )

    parser.add_argument(
        "--dataset",
        choices=["train", "test"],
        default="train",
    )

    parser.add_argument(
        "--methods",
        default="ascii,numeric,rare_name",
        help="Comma-separated methods",
    )

    args = parser.parse_args()

    EXPERIMENTS.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = TRAIN if args.dataset == "train" else TEST

    check_inputs(paths)

    methods = [
        x.strip().lower()
        for x in args.methods.split(",")
        if x.strip()
    ]

    allowed = {
        "ascii",
        "numeric",
        "rare_name",
    }

    invalid = set(methods) - allowed

    if invalid:
        raise ValueError(
            f"Unknown methods: {sorted(invalid)}"
        )

    print("=" * 80)
    print("RECALL EXPANSION EXPERIMENTS")
    print("=" * 80)
    print(f"Dataset : {args.dataset.upper()}")
    print(f"S1      : {paths['S1']}")
    print(f"S2      : {paths['S2']}")
    print(f"S3      : {paths['S3']}")
    print(f"Methods : {', '.join(methods)}")
    print("=" * 80)

    con = duckdb.connect()

    try:

        configure(con)

        # Required only for ASCII experiment.
        if "ascii" in methods:
            register_unidecode(con)

        for method in methods:

            if method == "ascii":

                write_ascii(
                    con,
                    args.dataset,
                    paths["S1"],
                    paths["S2"],
                    paths["S3"],
                )

            elif method == "numeric":

                write_numeric(
                    con,
                    args.dataset,
                    paths["S1"],
                    paths["S2"],
                    paths["S3"],
                )

            elif method == "rare_name":

                write_rare_name(
                    con,
                    args.dataset,
                    paths["S1"],
                    paths["S2"],
                    paths["S3"],
                )

        print("\n" + "=" * 80)
        print("RECALL EXPANSION COMPLETE")
        print("=" * 80)
        print(f"Experiment directory: {EXPERIMENTS}")

    finally:
        con.close()


if __name__ == "__main__":
    main()
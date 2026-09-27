from __future__ import annotations

"""
Amazon ML Challenge 2026: Indic Transliteration Candidate Generation.

Transliterates non-ASCII / Indic business names (Devanagari, Tamil, Telugu, etc.)
in Source 2 and Source 3 to Latin ASCII using unidecode, then blocks against
Source 1 businesses in India.

Features:
- Bit mask: 256
- Pure out-of-core streaming via DuckDB + pyarrow
- Memory ceiling <= 8GB
"""

import argparse
from pathlib import Path
import time
import duckdb
import pyarrow as pa
import unidecode

ROOT = Path(__file__).resolve().parents[2]
NORMALIZED_DIR = ROOT / "artifacts" / "normalized"
BLOCKING_DIR = ROOT / "artifacts" / "blocking"
CANDIDATE_DIR = BLOCKING_DIR / "candidates"
CANDIDATE_DIR.mkdir(parents=True, exist_ok=True)

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def build_indic_candidates(split: str = "train") -> Path:
    out_path = CANDIDATE_DIR / f"{split}_indic_translit_candidates.parquet"
    out_sql = sql_quote(out_path)

    s1_path = NORMALIZED_DIR / f"{split}_s1.parquet"
    s2_path = NORMALIZED_DIR / f"{split}_s2.parquet"
    s3_path = NORMALIZED_DIR / f"{split}_s3.parquet"

    print("\n" + "=" * 80)
    print(f"GENERATING INDIC TRANSLITERATION CANDIDATES ({split.upper()})")
    print("=" * 80)

    con = duckdb.connect()
    con.execute(f"SET threads = {THREADS}")
    con.execute(f"SET memory_limit = '{MEMORY_LIMIT}'")
    con.execute("SET preserve_insertion_order = false")

    try:
        # Transliterate S2 Indic names
        print(f"Transliterating S2 Indic names for {split}...")
        s2_reader = con.execute(f"""
            SELECT entity_id, name_norm, name_compact, country_norm
            FROM read_parquet('{sql_quote(s2_path)}')
            WHERE country_norm = 'india' AND strip_accents(name_norm) <> name_norm
        """).arrow()
        s2_table = s2_reader.read_all()

        s2_names = s2_table["name_norm"].to_pylist()
        s2_compact = s2_table["name_compact"].to_pylist()
        s2_t_names = [unidecode.unidecode(n).lower().strip() for n in s2_names]
        s2_t_compact = [unidecode.unidecode(c).lower().strip() for c in s2_compact]

        s2_with_translit = s2_table.append_column("name_translit", pa.array(s2_t_names)).append_column("name_compact_translit", pa.array(s2_t_compact))
        con.register("s2_indic", s2_with_translit)

        # Transliterate S3 Indic names
        print(f"Transliterating S3 Indic names for {split}...")
        s3_reader = con.execute(f"""
            SELECT entity_id, name_norm, name_compact, country_norm
            FROM read_parquet('{sql_quote(s3_path)}')
            WHERE country_norm = 'india' AND strip_accents(name_norm) <> name_norm
        """).arrow()
        s3_table = s3_reader.read_all()

        s3_names = s3_table["name_norm"].to_pylist()
        s3_compact = s3_table["name_compact"].to_pylist()
        s3_t_names = [unidecode.unidecode(n).lower().strip() for n in s3_names]
        s3_t_compact = [unidecode.unidecode(c).lower().strip() for c in s3_compact]

        s3_with_translit = s3_table.append_column("name_translit", pa.array(s3_t_names)).append_column("name_compact_translit", pa.array(s3_t_compact))
        con.register("s3_indic", s3_with_translit)

        # Block against S1 in India
        print("Matching transliterated names against S1...")
        query = f"""
        COPY (
            WITH s1 AS (
                SELECT entity_id, name_norm, name_compact, country_norm
                FROM read_parquet('{sql_quote(s1_path)}')
                WHERE country_norm = 'india' AND name_norm <> ''
            ),
            s2_matches AS (
                SELECT DISTINCT
                    s1.entity_id AS source1_entity_id,
                    t.entity_id AS matched_entity_id,
                    'S2' AS matched_source,
                    256 AS block_bit,
                    'indic_translit' AS block_name
                FROM s1
                INNER JOIN s2_indic t
                    ON s1.country_norm = t.country_norm
                   AND (s1.name_norm = t.name_translit OR s1.name_compact = t.name_compact_translit)
                   AND LENGTH(t.name_translit) >= 3
            ),
            s3_matches AS (
                SELECT DISTINCT
                    s1.entity_id AS source1_entity_id,
                    t.entity_id AS matched_entity_id,
                    'S3' AS matched_source,
                    256 AS block_bit,
                    'indic_translit' AS block_name
                FROM s1
                INNER JOIN s3_indic t
                    ON s1.country_norm = t.country_norm
                   AND (s1.name_norm = t.name_translit OR s1.name_compact = t.name_compact_translit)
                   AND LENGTH(t.name_translit) >= 3
            )
            SELECT * FROM s2_matches
            UNION ALL
            SELECT * FROM s3_matches
        ) TO '{out_sql}' (FORMAT PARQUET, COMPRESSION ZSTD);
        """
        con.execute(query)

        count = con.execute(f"SELECT COUNT(*) FROM read_parquet('{out_sql}')").fetchone()[0]
        print(f"Generated {count:,} Indic transliteration candidate pairs -> {out_path}")

    finally:
        con.close()

    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Indic transliteration candidates.")
    parser.add_argument("--split", choices=["train", "test", "both"], default="both")
    args = parser.parse_args()

    splits = ["train", "test"] if args.split == "both" else [args.split]
    for s in splits:
        build_indic_candidates(s)


if __name__ == "__main__":
    main()

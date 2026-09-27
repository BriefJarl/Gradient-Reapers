from __future__ import annotations

from pathlib import Path

import duckdb


import argparse

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "features"
)

DEFAULT_INPUT = OUTPUT_DIR / "train_phase3_features_labeled.parquet"
if not DEFAULT_INPUT.exists():
    DEFAULT_INPUT = OUTPUT_DIR / "train_features_labeled.parquet"

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def main() -> None:
    parser = argparse.ArgumentParser(description="Split labeled pairs into S1-leakage-free train and validation.")
    parser.add_argument("--input", type=str, default="", help="Input labeled parquet file.")
    parser.add_argument("--output-prefix", type=str, default="", help="Prefix for train/valid split outputs.")
    args = parser.parse_args()

    input_file = Path(args.input) if args.input else DEFAULT_INPUT
    prefix = args.output_prefix or ("phase3" if "phase3" in str(input_file) else "")

    train_out = OUTPUT_DIR / (f"train_{prefix}_split.parquet" if prefix else "train_split.parquet")
    valid_out = OUTPUT_DIR / (f"valid_{prefix}_split.parquet" if prefix else "valid_split.parquet")
    valid_s1_out = OUTPUT_DIR / (f"valid_{prefix}_s1_entities.parquet" if prefix else "valid_s1_entities.parquet")

    print("=" * 80)
    print("S1-LEVEL TRAIN / VALIDATION SPLIT")
    print("=" * 80)
    print(f"Input:    {input_file}")
    print(f"Train:    {train_out}")
    print(f"Valid:    {valid_out}")
    print(f"Valid S1: {valid_s1_out}")

    if not input_file.exists():
        raise FileNotFoundError(
            f"Input not found:\n{input_file}"
        )

    for path in (train_out, valid_out, valid_s1_out):
        if path.exists():
            path.unlink()

    con = duckdb.connect()

    try:

        con.execute(
            f"SET threads = {THREADS}"
        )

        con.execute(
            f"SET memory_limit = '{MEMORY_LIMIT}'"
        )

        con.execute(
            "SET preserve_insertion_order = false"
        )

        con.execute(
            "SET enable_progress_bar = true"
        )

        temp_dir = (
            ROOT
            / "artifacts"
            / "blocking"
            / "duckdb_tmp"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        con.execute(
            f"SET temp_directory = "
            f"'{sql_quote(temp_dir)}'"
        )

        input_sql = sql_quote(input_file)

        # ----------------------------------------------------
        # TRAIN
        # ----------------------------------------------------

        print()
        print("Creating TRAIN split...")

        con.execute(
            f"""
            COPY
            (
                SELECT *
                FROM read_parquet('{input_sql}')
                WHERE
                    MOD(
                        HASH(
                            source1_entity_id
                        ),
                        10
                    ) < 8
            )
            TO '{sql_quote(train_out)}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        # ----------------------------------------------------
        # VALID
        # ----------------------------------------------------

        print()
        print("Creating VALIDATION split...")

        con.execute(
            f"""
            COPY
            (
                SELECT *
                FROM read_parquet('{input_sql}')
                WHERE
                    MOD(
                        HASH(
                            source1_entity_id
                        ),
                        10
                    ) >= 8
            )
            TO '{sql_quote(valid_out)}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        # ----------------------------------------------------
        # DISTINCT VALID S1 ENTITIES
        # ----------------------------------------------------

        print()
        print("Extracting distinct validation S1 entities...")

        con.execute(
            f"""
            COPY
            (
                SELECT DISTINCT source1_entity_id
                FROM read_parquet('{sql_quote(valid_out)}')
            )
            TO '{sql_quote(valid_s1_out)}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY
            )
            """
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        train_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(train_out)}')
            """
        ).fetchone()[0]

        valid_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(valid_out)}')
            """
        ).fetchone()[0]

        overlap = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT DISTINCT source1_entity_id
                FROM read_parquet('{sql_quote(train_out)}')
            ) a

            INNER JOIN
            (
                SELECT DISTINCT source1_entity_id
                FROM read_parquet('{sql_quote(valid_out)}')
            ) b

            USING (source1_entity_id)
            """
        ).fetchone()[0]

        print()
        print("=" * 80)
        print("SPLIT SUMMARY")
        print("=" * 80)

        print(f"Train rows       : {train_count:,}")
        print(f"Validation rows  : {valid_count:,}")
        print(f"Total            : {train_count + valid_count:,}")
        print(f"S1 overlap       : {overlap:,}")

        if overlap != 0:
            raise RuntimeError(
                "S1 leakage detected between train and validation."
            )

        print()
        print("S1-level split check: PASS")

        print()
        print(f"TRAIN    : {train_out}")
        print(f"VALID    : {valid_out}")
        print(f"VALID S1 : {valid_s1_out}")

        print()
        print("=" * 80)
        print("TRAIN / VALIDATION SPLIT COMPLETE")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
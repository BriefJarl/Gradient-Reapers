from __future__ import annotations

from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

INPUT = (
    ROOT
    / "artifacts"
    / "features"
    / "train_features_labeled.parquet"
)

OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "features"
)

TRAIN = OUTPUT_DIR / "train_split.parquet"
VALID = OUTPUT_DIR / "valid_split.parquet"

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def main() -> None:

    print("=" * 80)
    print("S1-LEVEL TRAIN / VALIDATION SPLIT")
    print("=" * 80)

    if not INPUT.exists():
        raise FileNotFoundError(
            f"Input not found:\n{INPUT}"
        )

    for path in (TRAIN, VALID):
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

        input_sql = sql_quote(INPUT)

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
            TO '{sql_quote(TRAIN)}'
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
            TO '{sql_quote(VALID)}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        # ----------------------------------------------------
        # Counts
        # ----------------------------------------------------

        train_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(TRAIN)}')
            """
        ).fetchone()[0]

        valid_count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(VALID)}')
            """
        ).fetchone()[0]

        overlap = con.execute(
            f"""
            SELECT COUNT(*)
            FROM
            (
                SELECT DISTINCT source1_entity_id
                FROM read_parquet('{sql_quote(TRAIN)}')
            ) a

            INNER JOIN
            (
                SELECT DISTINCT source1_entity_id
                FROM read_parquet('{sql_quote(VALID)}')
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
        print(f"TRAIN : {TRAIN}")
        print(f"VALID : {VALID}")

        print()
        print("=" * 80)
        print("TRAIN / VALIDATION SPLIT COMPLETE")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
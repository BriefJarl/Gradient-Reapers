from __future__ import annotations

from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parents[2]

FEATURE_DIR = ROOT / "artifacts" / "features"

S2 = FEATURE_DIR / "train_features_s2.parquet"
S3 = FEATURE_DIR / "train_features_s3.parquet"
OUTPUT = FEATURE_DIR / "train_features.parquet"

THREADS = 8
MEMORY_LIMIT = "8GB"


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def sql_quote(path: Path) -> str:
    return sql_path(path).replace("'", "''")


def main() -> None:

    print("=" * 80)
    print("COMBINING TRAIN FEATURES")
    print("=" * 80)

    for path in (S2, S3):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing feature file:\n{path}"
            )

    if OUTPUT.exists():
        OUTPUT.unlink()

    con = duckdb.connect()

    try:

        con.execute(f"SET threads = {THREADS}")
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

        con.execute(
            f"""
            COPY
            (
                SELECT *
                FROM read_parquet(
                    [
                        '{sql_quote(S2)}',
                        '{sql_quote(S3)}'
                    ]
                )
            )
            TO '{sql_quote(OUTPUT)}'
            (
                FORMAT PARQUET,
                COMPRESSION SNAPPY,
                ROW_GROUP_SIZE 250000
            )
            """
        )

        count = con.execute(
            f"""
            SELECT COUNT(*)
            FROM read_parquet('{sql_quote(OUTPUT)}')
            """
        ).fetchone()[0]

        print()
        print(f"S2 rows : ", end="")
        print(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_quote(S2)}')
                """
            ).fetchone()[0]
        )

        print("S3 rows : ", end="")
        print(
            con.execute(
                f"""
                SELECT COUNT(*)
                FROM read_parquet('{sql_quote(S3)}')
                """
            ).fetchone()[0]
        )

        print(f"Final rows : {count:,}")

        print()
        print(f"Output:")
        print(OUTPUT)

        print()
        print("=" * 80)
        print("FEATURE COMBINATION COMPLETE")
        print("=" * 80)

    finally:
        con.close()


if __name__ == "__main__":
    main()
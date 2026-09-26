from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parents[2]
TRAIN_DIR = ROOT / "student_resource" / "dataset" / "train"


FILES = {
    "S1": TRAIN_DIR / "train_source1.tsv",
    "S2": TRAIN_DIR / "train_source2.tsv",
    "S3": TRAIN_DIR / "train_source3.tsv",
}


def sql_path(path: Path) -> str:
    return str(path).replace("\\", "/")


def profile_tokens(con, path: Path, source: str):

    print("\n" + "=" * 75)
    print(f"{source} NAME TOKEN FREQUENCY")
    print("=" * 75)

    p = sql_path(path)

    query = f"""
        WITH base AS (
            SELECT
                lower(
                    trim(
                        regexp_replace(
                            business_name,
                            '[^[:alnum:][:space:]]',
                            '',
                            'g'
                        )
                    )
                ) AS name_norm
            FROM read_csv_auto(
                '{p}',
                delim='\\t',
                header=true,
                sample_size=10000
            )
        ),
        tokens AS (
            SELECT
                unnest(
                    string_split(
                        name_norm,
                        ' '
                    )
                ) AS token
            FROM base
        )
        SELECT
            token,
            COUNT(*) AS frequency
        FROM tokens
        WHERE
            token <> ''
            AND length(token) >= 2
        GROUP BY token
        ORDER BY frequency DESC
        LIMIT 100
    """

    rows = con.execute(query).fetchall()

    for token, frequency in rows:
        print(f"{frequency:10,} | {token}")


def main():

    con = duckdb.connect()

    print("\nAMAZON ML CHALLENGE 2026")
    print("Token Frequency Profiler")

    for source, path in FILES.items():
        profile_tokens(con, path, source)

    print("\nToken profiling complete.")


if __name__ == "__main__":
    main()
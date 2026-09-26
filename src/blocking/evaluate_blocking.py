from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


ROOT = Path(__file__).resolve().parents[2]

TRAIN_DIR = (
    ROOT
    / "student_resource"
    / "dataset"
    / "train"
)

NORMALIZED_DIR = (
    ROOT
    / "artifacts"
    / "normalized"
)

BLOCKING_DIR = (
    ROOT
    / "artifacts"
    / "blocking"
)

CANDIDATE_DIR = (
    BLOCKING_DIR
    / "candidates"
)

UNION_DIR = (
    BLOCKING_DIR
    / "union"
)

GT_OUTPUT = (
    BLOCKING_DIR
    / "ground_truth_pairs.parquet"
)

GT_PATH = (
    TRAIN_DIR
    / "train_ground_truth.tsv"
)

S1_PATH = (
    NORMALIZED_DIR
    / "train_s1.parquet"
)

S2_PATH = (
    NORMALIZED_DIR
    / "train_s2.parquet"
)

S3_PATH = (
    NORMALIZED_DIR
    / "train_s3.parquet"
)


def sql_path(path: Path) -> str:

    return str(path).replace(
        "\\",
        "/",
    )


def resolve_candidate_file(
    filename: str,
) -> Path:

    supplied = Path(filename)

    # Allow an explicit relative/absolute path.
    if supplied.is_absolute():
        path = supplied

        if path.exists():
            return path

        raise FileNotFoundError(
            f"Candidate file not found:\n{path}"
        )

    # First: normal candidate directory.
    candidate_path = (
        CANDIDATE_DIR
        / filename
    )

    if candidate_path.exists():
        return candidate_path

    # Second: union directory.
    union_path = (
        UNION_DIR
        / filename
    )

    if union_path.exists():
        return union_path

    raise FileNotFoundError(
        "\nCandidate file not found in either location:\n"
        f"  candidates: {candidate_path}\n"
        f"  union     : {union_path}\n"
    )


def build_ground_truth(
    con: duckdb.DuckDBPyConnection,
) -> None:

    print(
        "\n" + "=" * 80
    )

    print(
        "BUILDING GROUND-TRUTH PAIR TABLE"
    )

    print(
        "=" * 80
    )

    gt = sql_path(GT_PATH)

    s2 = sql_path(S2_PATH)

    s3 = sql_path(S3_PATH)

    output = sql_path(GT_OUTPUT)

    overlap = con.execute(
        f"""
        WITH target_ids AS (

            SELECT
                entity_id,
                'S2' AS source

            FROM read_parquet(
                '{s2}'
            )

            UNION ALL

            SELECT
                entity_id,
                'S3' AS source

            FROM read_parquet(
                '{s3}'
            )
        )

        SELECT COUNT(*)

        FROM (

            SELECT
                entity_id

            FROM target_ids

            GROUP BY entity_id

            HAVING COUNT(
                DISTINCT source
            ) > 1
        )
        """
    ).fetchone()[0]

    print(
        f"Overlapping S2/S3 entity IDs: "
        f"{overlap:,}"
    )

    if overlap != 0:

        raise RuntimeError(
            "S2 and S3 entity IDs overlap."
        )

    query = f"""
        COPY (

            WITH gt_raw AS (

                SELECT
                    source1_entity_id,
                    matched_entity_ids

                FROM read_csv_auto(
                    '{gt}',
                    delim='\\t',
                    header=true,
                    sample_size=10000
                )
            ),

            exploded AS (

                SELECT

                    source1_entity_id,

                    TRIM(match_id)
                        AS matched_entity_id

                FROM gt_raw

                CROSS JOIN UNNEST(

                    string_split(

                        COALESCE(
                            matched_entity_ids,
                            ''
                        ),

                        ','
                    )

                ) AS t(match_id)

                WHERE TRIM(match_id) <> ''
            ),

            target_ids AS (

                SELECT

                    entity_id
                        AS matched_entity_id,

                    'S2'
                        AS matched_source

                FROM read_parquet(
                    '{s2}'
                )

                UNION ALL

                SELECT

                    entity_id
                        AS matched_entity_id,

                    'S3'
                        AS matched_source

                FROM read_parquet(
                    '{s3}'
                )
            )

            SELECT DISTINCT

                e.source1_entity_id,

                e.matched_entity_id,

                t.matched_source

            FROM exploded e

            INNER JOIN target_ids t

                ON e.matched_entity_id =
                   t.matched_entity_id

        )

        TO '{output}'

        (
            FORMAT PARQUET,
            COMPRESSION ZSTD
        );
    """

    con.execute(query)

    count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{output}'
        )
        """
    ).fetchone()[0]

    print(
        f"Ground-truth pairs: "
        f"{count:,}"
    )

    print(
        f"Output             : "
        f"{GT_OUTPUT}"
    )

def print_ground_truth_stats(
    con: duckdb.DuckDBPyConnection,
) -> None:

    gt = sql_path(
        GT_OUTPUT
    )

    s1 = sql_path(
        S1_PATH
    )

    print(
        "\n" + "=" * 80
    )

    print(
        "GROUND-TRUTH SOURCE DISTRIBUTION"
    )

    print(
        "=" * 80
    )

    rows = con.execute(
        f"""
        SELECT

            matched_source,

            COUNT(*) AS pair_count

        FROM read_parquet(
            '{gt}'
        )

        GROUP BY
            matched_source

        ORDER BY
            matched_source
        """
    ).fetchall()

    for source, count in rows:

        print(
            f"{source}: "
            f"{count:,} true pairs"
        )

    print(
        "\nMatches per Source-1 entity:"
    )

    distribution = con.execute(
        f"""
        WITH match_counts AS (

            SELECT

                s1.entity_id
                    AS source1_entity_id,

                COUNT(
                    gt.matched_entity_id
                ) AS match_count

            FROM read_parquet(
                '{s1}'
            ) s1

            LEFT JOIN read_parquet(
                '{gt}'
            ) gt

                ON s1.entity_id =
                   gt.source1_entity_id

            GROUP BY
                s1.entity_id
        )

        SELECT

            match_count,

            COUNT(*) AS entity_count

        FROM match_counts

        GROUP BY
            match_count

        ORDER BY
            match_count
        """
    ).fetchall()

    for match_count, entity_count in distribution:

        print(
            f"{match_count:3} matches -> "
            f"{entity_count:,} S1 entities"
        )


def evaluate_candidates(
    con: duckdb.DuckDBPyConnection,
    candidate_path: Path,
) -> None:

    candidate = sql_path(
        candidate_path
    )

    gt = sql_path(
        GT_OUTPUT
    )

    s1 = sql_path(
        S1_PATH
    )

    print(
        "\n" + "=" * 80
    )

    print(
        f"EVALUATING: "
        f"{candidate_path}"
    )

    print(
        "=" * 80
    )

   
    candidate_count = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{candidate}'
        )
        """
    ).fetchone()[0]

    unique_candidate_count = con.execute(
        f"""
        SELECT COUNT(*)

        FROM (

            SELECT DISTINCT

                source1_entity_id,

                matched_entity_id,

                matched_source

            FROM read_parquet(
                '{candidate}'
            )
        )
        """
    ).fetchone()[0]

    print(
        f"Candidate pairs : "
        f"{candidate_count:,}"
    )

    print(
        f"Unique pairs    : "
        f"{unique_candidate_count:,}"
    )

    
    # Ground-truth recovery.
    recovered = con.execute(
        f"""
        SELECT COUNT(*)

        FROM (

            SELECT DISTINCT

                c.source1_entity_id,

                c.matched_entity_id,

                c.matched_source

            FROM read_parquet(
                '{candidate}'
            ) c

            INNER JOIN read_parquet(
                '{gt}'
            ) g

                ON c.source1_entity_id =
                   g.source1_entity_id

                AND c.matched_entity_id =
                    g.matched_entity_id

                AND c.matched_source =
                    g.matched_source
        )
        """
    ).fetchone()[0]

    total_true = con.execute(
        f"""
        SELECT COUNT(*)
        FROM read_parquet(
            '{gt}'
        )
        """
    ).fetchone()[0]

    recall = (
        recovered / total_true
        if total_true
        else 0.0
    )

    purity = (
        recovered
        / unique_candidate_count
        if unique_candidate_count
        else 0.0
    )

    print(
        f"True pairs recovered : "
        f"{recovered:,}"
    )

    print(
        f"Total true pairs    : "
        f"{total_true:,}"
    )

    print(
        f"Candidate recall    : "
        f"{recall:.6%}"
    )

    print(
        f"Candidate purity    : "
        f"{purity:.6%}"
    )

    
    # Source-wise recall.
    print(
        "\nSource-wise recall:"
    )

    source_rows = con.execute(
        f"""
        WITH totals AS (

            SELECT

                matched_source,

                COUNT(*) AS total_pairs

            FROM read_parquet(
                '{gt}'
            )

            GROUP BY
                matched_source
        ),

        found AS (

            SELECT DISTINCT

                c.source1_entity_id,

                c.matched_entity_id,

                c.matched_source

            FROM read_parquet(
                '{candidate}'
            ) c

            INNER JOIN read_parquet(
                '{gt}'
            ) g

                ON c.source1_entity_id =
                   g.source1_entity_id

                AND c.matched_entity_id =
                    g.matched_entity_id

                AND c.matched_source =
                    g.matched_source
        )

        SELECT

            t.matched_source,

            t.total_pairs,

            COUNT(
                f.matched_entity_id
            ) AS recovered,

            COUNT(
                f.matched_entity_id
            ) * 1.0
            / t.total_pairs
                AS recall

        FROM totals t

        LEFT JOIN found f

            ON t.matched_source =
               f.matched_source

        GROUP BY

            t.matched_source,

            t.total_pairs

        ORDER BY
            t.matched_source
        """
    ).fetchall()

    for (
        source,
        total,
        found,
        source_recall,
    ) in source_rows:

        print(
            f"{source}: "
            f"{found:,}/{total:,} "
            f"({source_recall:.6%})"
        )

    # Candidate distribution per S1.
    print(
        "\nCandidates per Source-1 entity:"
    )

    distribution = con.execute(
        f"""
        WITH candidate_counts AS (

            SELECT

                source1_entity_id,

                COUNT(*) AS candidate_count

            FROM read_parquet(
                '{candidate}'
            )

            GROUP BY
                source1_entity_id
        ),

        all_s1 AS (

            SELECT

                s1.entity_id
                    AS source1_entity_id,

                COALESCE(
                    c.candidate_count,
                    0
                ) AS candidate_count

            FROM read_parquet(
                '{s1}'
            ) s1

            LEFT JOIN candidate_counts c

                ON s1.entity_id =
                   c.source1_entity_id
        )

        SELECT

            AVG(candidate_count),

            quantile_cont(
                candidate_count,
                0.50
            ),

            quantile_cont(
                candidate_count,
                0.90
            ),

            quantile_cont(
                candidate_count,
                0.95
            ),

            quantile_cont(
                candidate_count,
                0.99
            ),

            MAX(candidate_count)

        FROM all_s1
        """
    ).fetchone()

    (
        mean_candidates,
        p50,
        p90,
        p95,
        p99,
        max_candidates,
    ) = distribution

    print(
        f"Mean : {mean_candidates:.3f}"
    )

    print(
        f"P50  : {p50:.3f}"
    )

    print(
        f"P90  : {p90:.3f}"
    )

    print(
        f"P95  : {p95:.3f}"
    )

    print(
        f"P99  : {p99:.3f}"
    )

    print(
        f"MAX  : {max_candidates:,}"
    )
    
    # Candidate provenance, when available.
    
    columns = con.execute(
        f"""
        DESCRIBE
        SELECT *
        FROM read_parquet(
            '{candidate}'
        )
        """
    ).fetchall()

    column_names = {
        row[0]
        for row in columns
    }

    if "num_blocking_methods" in column_names:

        print(
            "\nBlocking provenance:"
        )

        provenance = con.execute(
            f"""
            SELECT

                num_blocking_methods,

                COUNT(*) AS pair_count

            FROM read_parquet(
                '{candidate}'
            )

            GROUP BY
                num_blocking_methods

            ORDER BY
                num_blocking_methods
            """
        ).fetchall()

        for (
            method_count,
            pair_count,
        ) in provenance:

            print(
                f"  {method_count} method(s): "
                f"{pair_count:,}"
            )


def main() -> None:

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--candidates",
        required=True,
        help=(
            "Candidate parquet filename. "
            "The evaluator searches both "
            "artifacts/blocking/candidates "
            "and artifacts/blocking/union."
        ),
    )

    args = parser.parse_args()

    candidate_path = (
        resolve_candidate_file(
            args.candidates
        )
    )

    con = duckdb.connect()

    try:

        temp_dir = (
            BLOCKING_DIR
            / "tmp"
        )

        temp_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        con.execute(
            f"""
            SET temp_directory =
                '{sql_path(temp_dir)}';
            """
        )

        con.execute(
            """
            SET threads = 8;
            """
        )

        if not GT_OUTPUT.exists():

            build_ground_truth(
                con
            )

        print_ground_truth_stats(
            con
        )

        evaluate_candidates(
            con,
            candidate_path,
        )

    finally:

        con.close()

    print(
        "\n" + "=" * 80
    )

    print(
        "BLOCKING EVALUATION COMPLETE"
    )

    print(
        "=" * 80
    )


if __name__ == "__main__":

    main()

from pathlib import Path
import duckdb


ROOT = Path(__file__).resolve().parents[2]

CANDIDATE_PATH = (
    ROOT
    / "artifacts"
    / "blocking"
    / "union"
    / "train_exact_rare_address_union.parquet"
)

GROUND_TRUTH_PATH = (
    ROOT
    / "artifacts"
    / "blocking"
    / "ground_truth_pairs.parquet"
)

NORMALIZED_DIR = ROOT / "artifacts" / "normalized"


def main():

    print("=" * 80)
    print("CANDIDATE QUALITY ANALYSIS")
    print("=" * 80)

    con = duckdb.connect()

    con.execute("SET threads = 8")

    print("\n" + "-" * 80)
    print("BLOCKING MASK DISTRIBUTION")
    print("-" * 80)

    mask_df = con.execute(
        f"""
        SELECT
            blocking_mask,
            blocking_methods,
            COUNT(*) AS candidate_pairs
        FROM read_parquet('{CANDIDATE_PATH}')
        GROUP BY
            blocking_mask,
            blocking_methods
        ORDER BY candidate_pairs DESC
        """
    ).fetchdf()

    print(mask_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("RARE-ADDRESS-ONLY CANDIDATES")
    print("-" * 80)

    rare_address_df = con.execute(
        f"""
        SELECT COUNT(*) AS count
        FROM read_parquet('{CANDIDATE_PATH}')
        WHERE blocking_mask = 16
        """
    ).fetchdf()

    print(rare_address_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("TRUE PAIRS BY BLOCKING MASK")
    print("-" * 80)

    truth_df = con.execute(
        f"""
        SELECT
            c.blocking_mask,
            c.blocking_methods,
            COUNT(*) AS true_pairs
        FROM read_parquet('{CANDIDATE_PATH}') c
        INNER JOIN read_parquet('{GROUND_TRUTH_PATH}') g
            ON c.source1_entity_id = g.source1_entity_id
            AND c.matched_entity_id = g.matched_entity_id
            AND c.matched_source = g.matched_source
        GROUP BY
            c.blocking_mask,
            c.blocking_methods
        ORDER BY true_pairs DESC
        """
    ).fetchdf()

    print(truth_df.to_string(index=False))

    print("\n" + "-" * 80)
    print("CANDIDATE QUALITY BY BLOCKING MASK")
    print("-" * 80)

    quality_df = con.execute(
        f"""
        WITH candidates AS (
            SELECT
                blocking_mask,
                blocking_methods,
                COUNT(*) AS candidates
            FROM read_parquet('{CANDIDATE_PATH}')
            GROUP BY
                blocking_mask,
                blocking_methods
        ),
        truths AS (
            SELECT
                c.blocking_mask,
                COUNT(*) AS true_pairs
            FROM read_parquet('{CANDIDATE_PATH}') c
            INNER JOIN read_parquet('{GROUND_TRUTH_PATH}') g
                ON c.source1_entity_id = g.source1_entity_id
                AND c.matched_entity_id = g.matched_entity_id
                AND c.matched_source = g.matched_source
            GROUP BY c.blocking_mask
        )
        SELECT
            c.blocking_mask,
            c.blocking_methods,
            c.candidates,
            COALESCE(t.true_pairs, 0) AS true_pairs,
            ROUND(
                100.0 * COALESCE(t.true_pairs, 0)
                / NULLIF(c.candidates, 0),
                4
            ) AS purity_percent
        FROM candidates c
        LEFT JOIN truths t
            ON c.blocking_mask = t.blocking_mask
        ORDER BY c.candidates DESC
        """
    ).fetchdf()

    print(quality_df.to_string(index=False))

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()

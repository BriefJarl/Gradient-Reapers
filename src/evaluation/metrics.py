from __future__ import annotations

"""
Evaluation metrics for Amazon ML Challenge 2026: Business Entity Resolution.

Official metric: Macro-averaged F_0.5 score across all Source 1 entities in the evaluation set.
F_0.5 formula:
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    = (1.25 * TP) / (Predicted_Count + 0.25 * True_Count)

Singletons rule:
    - If true matches == 0 and predicted matches == 0: Score = 1.0
    - If true matches == 0 and predicted matches > 0: Score = 0.0
    - If true matches > 0 and predicted matches == 0: Score = 0.0
"""

from pathlib import Path
from typing import Mapping, Sequence, Set
import duckdb


def compute_entity_f05(
    predicted_matches: Set[str],
    true_matches: Set[str],
) -> tuple[float, float, float]:
    """
    Compute Precision, Recall, and F_0.5 for a single Source 1 entity.

    Returns:
        (f05, precision, recall)
    """
    num_pred = len(predicted_matches)
    num_true = len(true_matches)

    # Singleton correctly predicted as empty
    if num_pred == 0 and num_true == 0:
        return 1.0, 1.0, 1.0

    # Singleton false positive (predicted match when none exists)
    if num_true == 0 and num_pred > 0:
        return 0.0, 0.0, 1.0

    # Missed match (predicted empty when true matches exist)
    if num_pred == 0 and num_true > 0:
        return 0.0, 0.0, 0.0

    # Both non-empty
    true_positives = len(predicted_matches.intersection(true_matches))

    if true_positives == 0:
        return 0.0, 0.0, 0.0

    precision = true_positives / num_pred
    recall = true_positives / num_true

    f05 = (1.25 * true_positives) / (num_pred + 0.25 * num_true)

    return f05, precision, recall


def compute_macro_metrics(
    predictions: Mapping[str, Set[str]],
    ground_truth: Mapping[str, Set[str]],
    all_s1_ids: Sequence[str] | Set[str],
) -> dict[str, float]:
    """
    Compute Macro-averaged metrics across all Source 1 entities in the evaluation set.

    Every Source 1 entity in all_s1_ids MUST be included.
    """
    total_entities = len(all_s1_ids)
    if total_entities == 0:
        return {
            "macro_f05": 0.0,
            "macro_precision": 0.0,
            "macro_recall": 0.0,
            "singleton_accuracy": 0.0,
            "total_entities": 0,
        }

    sum_f05 = 0.0
    sum_precision = 0.0
    sum_recall = 0.0

    singleton_count = 0
    singleton_correct = 0

    for s1_id in all_s1_ids:
        preds = predictions.get(s1_id, set())
        trues = ground_truth.get(s1_id, set())

        f05, prec, rec = compute_entity_f05(preds, trues)

        sum_f05 += f05
        sum_precision += prec
        sum_recall += rec

        if len(trues) == 0:
            singleton_count += 1
            if len(preds) == 0:
                singleton_correct += 1

    singleton_acc = (
        singleton_correct / singleton_count if singleton_count > 0 else 1.0
    )

    return {
        "macro_f05": sum_f05 / total_entities,
        "macro_precision": sum_precision / total_entities,
        "macro_recall": sum_recall / total_entities,
        "singleton_accuracy": singleton_acc,
        "total_entities": float(total_entities),
    }


def sql_quote(path: Path | str) -> str:
    return str(path).replace("\\", "/").replace("'", "''")


def evaluate_predictions_duckdb(
    con: duckdb.DuckDBPyConnection,
    predictions_query_or_table: str,
    ground_truth_path: Path,
    s1_split_path: Path,
) -> dict[str, float]:
    """
    Fast vectorized Macro F_0.5 evaluation in DuckDB across millions of pairs.

    Expects `predictions_query_or_table` to yield rows:
        (source1_entity_id, matched_entity_id)

    `ground_truth_path` should be a Parquet table with:
        (source1_entity_id, matched_entity_id)

    `s1_split_path` should be a Parquet table containing:
        (source1_entity_id)
    representing all S1 entities that must be evaluated.
    """
    gt_sql = sql_quote(ground_truth_path)
    s1_sql = sql_quote(s1_split_path)

    query = f"""
    WITH all_s1 AS (
        SELECT DISTINCT source1_entity_id
        FROM read_parquet('{s1_sql}')
    ),

    preds AS (
        SELECT DISTINCT source1_entity_id, matched_entity_id
        FROM ({predictions_query_or_table})
    ),

    trues AS (
        SELECT DISTINCT source1_entity_id, matched_entity_id
        FROM read_parquet('{gt_sql}')
    ),

    pred_counts AS (
        SELECT source1_entity_id, COUNT(matched_entity_id) AS num_pred
        FROM preds
        GROUP BY source1_entity_id
    ),

    true_counts AS (
        SELECT source1_entity_id, COUNT(matched_entity_id) AS num_true
        FROM trues
        GROUP BY source1_entity_id
    ),

    tp_counts AS (
        SELECT p.source1_entity_id, COUNT(*) AS num_tp
        FROM preds p
        INNER JOIN trues t
            ON p.source1_entity_id = t.source1_entity_id
           AND p.matched_entity_id = t.matched_entity_id
        GROUP BY p.source1_entity_id
    ),

    entity_metrics AS (
        SELECT
            s.source1_entity_id,
            COALESCE(p.num_pred, 0) AS num_pred,
            COALESCE(t.num_true, 0) AS num_true,
            COALESCE(tp.num_tp, 0) AS num_tp,
            CASE
                -- Singleton correct: both 0
                WHEN COALESCE(p.num_pred, 0) = 0 AND COALESCE(t.num_true, 0) = 0 THEN 1.0
                -- Singleton false merge
                WHEN COALESCE(t.num_true, 0) = 0 AND COALESCE(p.num_pred, 0) > 0 THEN 0.0
                -- Missed matches
                WHEN COALESCE(p.num_pred, 0) = 0 AND COALESCE(t.num_true, 0) > 0 THEN 0.0
                -- True positives exist
                WHEN COALESCE(tp.num_tp, 0) > 0 THEN
                    (1.25 * CAST(tp.num_tp AS DOUBLE)) / (CAST(p.num_pred AS DOUBLE) + 0.25 * CAST(t.num_true AS DOUBLE))
                ELSE 0.0
            END AS f05,
            CASE
                WHEN COALESCE(p.num_pred, 0) = 0 AND COALESCE(t.num_true, 0) = 0 THEN 1.0
                WHEN COALESCE(p.num_pred, 0) = 0 THEN 0.0
                ELSE CAST(COALESCE(tp.num_tp, 0) AS DOUBLE) / CAST(p.num_pred AS DOUBLE)
            END AS precision,
            CASE
                WHEN COALESCE(p.num_pred, 0) = 0 AND COALESCE(t.num_true, 0) = 0 THEN 1.0
                WHEN COALESCE(t.num_true, 0) = 0 THEN 1.0
                ELSE CAST(COALESCE(tp.num_tp, 0) AS DOUBLE) / CAST(t.num_true AS DOUBLE)
            END AS recall
        FROM all_s1 s
        LEFT JOIN pred_counts p ON s.source1_entity_id = p.source1_entity_id
        LEFT JOIN true_counts t ON s.source1_entity_id = t.source1_entity_id
        LEFT JOIN tp_counts tp ON s.source1_entity_id = tp.source1_entity_id
    )

    SELECT
        AVG(f05) AS macro_f05,
        AVG(precision) AS macro_precision,
        AVG(recall) AS macro_recall,
        COUNT(*) AS total_entities,
        SUM(CASE WHEN num_true = 0 THEN 1 ELSE 0 END) AS total_singletons,
        SUM(CASE WHEN num_true = 0 AND num_pred = 0 THEN 1 ELSE 0 END) AS correct_singletons
    FROM entity_metrics
    """

    res = con.execute(query).fetchone()

    macro_f05, macro_prec, macro_rec, total_ents, total_sings, corr_sings = res
    sing_acc = corr_sings / total_sings if total_sings > 0 else 1.0

    return {
        "macro_f05": float(macro_f05 or 0.0),
        "macro_precision": float(macro_prec or 0.0),
        "macro_recall": float(macro_rec or 0.0),
        "singleton_accuracy": float(sing_acc),
        "total_entities": float(total_ents),
        "total_singletons": float(total_sings),
        "correct_singletons": float(corr_sings),
    }

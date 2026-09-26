from __future__ import annotations

"""
Production TEST scorer
Loads the frozen LightGBM model and scores TEST features in Arrow batches.
No pandas and no full TEST matrix in RAM.
"""

import argparse
import json
import os
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
FEATURES = ROOT / "artifacts" / "features" / "final_test" / "test_features.parquet"
MODEL = ROOT / "artifacts" / "models" / "final_lightgbm.txt"
META = ROOT / "artifacts" / "models" / "final_lightgbm_meta.json"
OUT = ROOT / "artifacts" / "features" / "final_test" / "test_predictions.parquet"

BATCH_SIZE = int(os.environ.get("PREDICT_BATCH", "250000"))

MODEL_FEATURES = [
    "blocking_mask", "num_blocking_methods",
    "name_exact", "name_compact_exact", "name_ascii_exact",
    "name_char_ratio", "name_token_overlap", "name_token_jaccard",
    "name_length_diff", "name_token_count_diff", "name_numeric_overlap",
    "name_numeric_exact", "name_length_ratio",
    "address_exact", "address_compact_exact", "address_ascii_exact",
    "address_char_ratio", "address_token_overlap", "address_token_jaccard",
    "address_length_diff", "address_token_count_diff",
    "address_numeric_exact", "address_numeric_overlap", "address_length_ratio",
    "country_exact", "name_present", "address_present",
    "block_address", "block_address_compact", "block_name",
    "block_rare_name", "block_rare_address", "block_hybrid",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    for p in (FEATURES, MODEL):
        if not p.exists():
            raise FileNotFoundError(p)

    if OUT.exists() and not args.force:
        print(f"Existing prediction file found: {OUT}")
        print("Delete it or use --force to rescore.")
        return

    booster = lgb.Booster(model_file=str(MODEL))

    # Metadata is advisory; the explicit feature contract is authoritative.
    if META.exists():
        meta = json.loads(META.read_text(encoding="utf-8"))
        saved = meta.get("model_features") or meta.get("features")
        if saved and list(saved) != MODEL_FEATURES:
            raise RuntimeError(
                "Model feature contract mismatch between metadata and scorer."
            )

    parquet = pq.ParquetFile(FEATURES)
    total = parquet.metadata.num_rows

    writer = None
    partial = OUT.with_suffix(".partial.parquet")
    if partial.exists():
        partial.unlink()

    processed = 0
    try:
        for batch in parquet.iter_batches(
            batch_size=BATCH_SIZE,
            columns=[
                "source1_entity_id",
                "matched_entity_id",
                "matched_source",
                *MODEL_FEATURES,
            ],
        ):
            data = batch.to_pydict()
            X = np.column_stack([
                np.asarray(data[c], dtype=np.float32)
                for c in MODEL_FEATURES
            ])

            scores = booster.predict(X)
            scores = np.asarray(scores, dtype=np.float32)

            table = pa.table({
                "source1_entity_id": pa.array(data["source1_entity_id"]),
                "matched_entity_id": pa.array(data["matched_entity_id"]),
                "matched_source": pa.array(data["matched_source"]),
                "score": pa.array(scores, type=pa.float32()),
            })

            if writer is None:
                writer = pq.ParquetWriter(
                    str(partial),
                    table.schema,
                    compression="zstd",
                )

            writer.write_table(table)
            processed += len(scores)

            print(
                f"\rScored {processed:,}/{total:,} "
                f"({processed / total * 100:.1f}%)",
                end="",
                flush=True,
            )
    finally:
        if writer is not None:
            writer.close()

    print()
    if processed != total:
        raise RuntimeError(f"Prediction row mismatch: {processed:,} != {total:,}")

    if OUT.exists():
        OUT.unlink()
    partial.replace(OUT)

    print("=" * 88)
    print("TEST PREDICTION BUILD PASSED")
    print("=" * 88)
    print(f"Rows  : {processed:,}")
    print(f"Output: {OUT}")


if __name__ == "__main__":
    main()

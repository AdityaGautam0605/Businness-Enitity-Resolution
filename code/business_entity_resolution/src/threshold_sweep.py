import csv
from pathlib import Path

import numpy as np
import pandas as pd


SCORES_PATH = Path("output/baseline-aws-resume-01/pair_scores.tsv")
TRUTH_PATH = Path("dataset/train_ground_truth.tsv")
OUTPUT_PATH = Path("output/baseline-aws-resume-01/threshold_sweep.tsv")

CHUNK_SIZE = 1_000_000

THRESHOLDS = np.round(
    np.arange(0.8000, 0.8041, 0.0001),
    4,
)


def load_truth():
    print("Loading ground truth...", flush=True)

    truth = {}

    with open(
        TRUTH_PATH,
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        reader = csv.DictReader(
            handle,
            delimiter="\t",
        )

        for row in reader:
            source = row["source1_entity_id"]
            packed = row["matched_entity_ids"]

            if packed:
                truth[source] = set(
                    target
                    for target in packed.split(",")
                    if target
                )
            else:
                truth[source] = set()

    print(
        f"Ground-truth entities: {len(truth):,}",
        flush=True,
    )

    return truth


def process_chunk(
    df,
    truth,
    macro_sum,
    micro_tp,
    micro_pred,
):
    if df.empty:
        return 0, 0

    source_codes, source_ids = pd.factorize(
        df["source1_entity_id"],
        sort=False,
    )

    source_truth_sets = [
        truth.get(source, set())
        for source in source_ids
    ]

    truth_counts = np.fromiter(
        (
            len(source_truth)
            for source_truth in source_truth_sets
        ),
        dtype=np.int32,
        count=len(source_ids),
    )

    source_truth_lookup = {
        source: truth_set
        for source, truth_set in zip(
            source_ids,
            source_truth_sets,
        )
    }

    is_true = np.fromiter(
        (
            target
            in source_truth_lookup.get(
                source,
                ()
            )
            for source, target in zip(
                df["source1_entity_id"],
                df["target_entity_id"],
            )
        ),
        dtype=np.bool_,
        count=len(df),
    )

    scores = df["score"].to_numpy(
        dtype=np.float32,
        copy=False,
    )

    threshold_bins = np.searchsorted(
        THRESHOLDS,
        scores,
        side="right",
    )

    k = len(THRESHOLDS)
    n_sources = len(source_ids)

    flat_index = (
        source_codes * (k + 1)
        + threshold_bins
    )

    counts = np.bincount(
        flat_index,
        minlength=n_sources * (k + 1),
    ).reshape(
        n_sources,
        k + 1,
    )

    true_flat_index = flat_index[
        is_true
    ]

    true_counts = np.bincount(
        true_flat_index,
        minlength=n_sources * (k + 1),
    ).reshape(
        n_sources,
        k + 1,
    )

    cumulative_counts = np.cumsum(
        counts[:, ::-1],
        axis=1,
    )[:, ::-1]

    cumulative_true = np.cumsum(
        true_counts[:, ::-1],
        axis=1,
    )[:, ::-1]

    predicted = cumulative_counts[:, 1:]
    true_positive = cumulative_true[:, 1:]

    micro_pred += predicted.sum(
        axis=0,
    )

    micro_tp += true_positive.sum(
        axis=0,
    )

    positive_truth = (
        truth_counts > 0
    )

    if positive_truth.any():
        tc = truth_counts[
            positive_truth
        ][:, None]

        pred = predicted[
            positive_truth
        ]

        tp = true_positive[
            positive_truth
        ]

        denominator = (
            0.25 * tc + pred
        )

        scores_f05 = np.divide(
            1.25 * tp,
            denominator,
            out=np.zeros_like(
                denominator,
                dtype=np.float64,
            ),
            where=denominator != 0,
        )

        macro_sum += scores_f05.sum(
            axis=0
        )

    singleton_truth = ~positive_truth

    if singleton_truth.any():
        singleton_predictions = predicted[
            singleton_truth
        ]

        macro_sum += (
            singleton_predictions == 0
        ).sum(
            axis=0
        )

    singleton_count = int(
        singleton_truth.sum()
    )

    return n_sources, singleton_count


def main():
    truth = load_truth()

    total_truth_pairs = sum(
        len(ids)
        for ids in truth.values()
    )

    total_entities = len(truth)

    total_singletons = sum(
        len(ids) == 0
        for ids in truth.values()
    )

    print(
        f"True pairs: {total_truth_pairs:,}",
        flush=True,
    )

    print(
        f"Singleton entities: {total_singletons:,}",
        flush=True,
    )

    k = len(THRESHOLDS)

    macro_sum = np.zeros(
        k,
        dtype=np.float64,
    )

    micro_tp = np.zeros(
        k,
        dtype=np.int64,
    )

    micro_pred = np.zeros(
        k,
        dtype=np.int64,
    )

    processed_rows = 0
    processed_sources = 0
    seen_singletons = 0

    carry = None

    reader = pd.read_csv(
        SCORES_PATH,
        sep="\t",
        usecols=[
            "source1_entity_id",
            "target_entity_id",
            "score",
        ],
        dtype={
            "source1_entity_id": str,
            "target_entity_id": str,
            "score": np.float32,
        },
        chunksize=CHUNK_SIZE,
    )

    for chunk_number, chunk in enumerate(
        reader,
        start=1,
    ):
        if carry is not None:
            chunk = pd.concat(
                [carry, chunk],
                ignore_index=True,
            )

        last_source = chunk[
            "source1_entity_id"
        ].iloc[-1]

        last_mask = (
            chunk["source1_entity_id"]
            == last_source
        )

        carry = chunk.loc[
            last_mask
        ].copy()

        process_df = chunk.loc[
            ~last_mask
        ]

        source_count, singleton_count = (
            process_chunk(
                process_df,
                truth,
                macro_sum,
                micro_tp,
                micro_pred,
            )
        )

        processed_rows += len(
            process_df
        )

        processed_sources += (
            source_count
        )

        seen_singletons += (
            singleton_count
        )

        print(
            f"Chunk {chunk_number}: "
            f"{processed_rows:,} scored pairs, "
            f"{processed_sources:,} sources",
            flush=True,
        )

    if carry is not None:
        source_count, singleton_count = (
            process_chunk(
                carry,
                truth,
                macro_sum,
                micro_tp,
                micro_pred,
            )
        )

        processed_rows += len(carry)
        processed_sources += source_count
        seen_singletons += singleton_count

    # Sources with no candidate pairs never appeared in pair_scores.
    # Positive-truth entities contribute F0.5 = 0.
    # Singleton entities with zero predictions contribute F0.5 = 1.
    missing_singletons = (
        total_singletons
        - seen_singletons
    )

    macro_sum += missing_singletons

    false_positive = (
        micro_pred - micro_tp
    )

    false_negative = (
        total_truth_pairs - micro_tp
    )

    precision = np.divide(
        micro_tp,
        micro_pred,
        out=np.zeros(
            k,
            dtype=np.float64,
        ),
        where=micro_pred != 0,
    )

    recall = (
        micro_tp / total_truth_pairs
    )

    macro_f05 = (
        macro_sum / total_entities
    )

    results = pd.DataFrame(
        {
            "threshold": THRESHOLDS,
            "macro_f05": macro_f05,
            "micro_precision": precision,
            "micro_recall": recall,
            "true_positives": micro_tp,
            "false_positives": false_positive,
            "false_negatives": false_negative,
            "predicted_pairs": micro_pred,
        }
    )

    results = results.sort_values(
        "macro_f05",
        ascending=False,
    ).reset_index(
        drop=True
    )

    results.to_csv(
        OUTPUT_PATH,
        sep="\t",
        index=False,
    )

    print("\nTOP THRESHOLDS", flush=True)

    print(
        results.head(10).to_string(
            index=False
        ),
        flush=True,
    )

    print(
        f"\nSaved: {OUTPUT_PATH}",
        flush=True,
    )


if __name__ == "__main__":
    main()
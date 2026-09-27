from pathlib import Path
import time

import pandas as pd

from aggregate import aggregate_to_tsv_format
from blocking import PAIR_COLUMNS
from config import PipelineConfig
from features import compute_pairwise_features
from model import score_pairs


INPUT_DIR = Path("output/baseline-aws-01")
OUTPUT_DIR = Path("output/baseline-aws-resume-01")

SOURCE_ROWS_PER_BATCH = 2500
FEATURE_INNER_CHUNK = 100000

SAVE_FEATURES = False
SAVE_SCORES = True


def append_tsv(df, path):
    df.to_csv(
        path,
        sep="\t",
        index=False,
        mode="a",
        header=not path.exists(),
    )


def expand_candidate_chunk(chunk):
    source_ids = []
    target_ids = []

    for source_id, packed_targets in chunk[
        ["source1_entity_id", "candidate_entity_ids"]
    ].itertuples(index=False, name=None):

        if not packed_targets:
            continue

        targets = packed_targets.split(",")

        for target_id in targets:
            target_id = target_id.strip()

            if not target_id:
                continue

            source_ids.append(source_id)
            target_ids.append(target_id)

    return pd.DataFrame(
        {
            "source1_entity_id": source_ids,
            "target_entity_id": target_ids,
        }
    )


def main():
    source_path = INPUT_DIR / "normalized_source1.tsv"
    target_path = INPUT_DIR / "normalized_targets.tsv"
    candidate_path = INPUT_DIR / "candidate_pairs.tsv"

    if not source_path.exists():
        raise FileNotFoundError(source_path)

    if not target_path.exists():
        raise FileNotFoundError(target_path)

    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    feature_path = OUTPUT_DIR / "pair_features.tsv"
    score_path = OUTPUT_DIR / "pair_scores.tsv"
    result_path = OUTPUT_DIR / "matching_results.tsv"

    for path in (
        feature_path,
        score_path,
        result_path,
    ):
        if path.exists():
            raise RuntimeError(
                f"{path} already exists. "
                "Use a new output directory or remove it first."
            )

    print("Loading normalized Source 1...", flush=True)

    source_norm = pd.read_csv(
        source_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Loaded {len(source_norm):,} source rows",
        flush=True,
    )

    print("Loading normalized targets...", flush=True)

    target_norm = pd.read_csv(
        target_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Loaded {len(target_norm):,} target rows",
        flush=True,
    )

    print("Building ID lookup indexes...", flush=True)

    source_index = pd.Index(
        source_norm["entity_id"],
        copy=False,
    )

    target_index = pd.Index(
        target_norm["entity_id"],
        copy=False,
    )

    config = PipelineConfig()

    threshold = config.baseline_threshold

    print(
        f"Baseline threshold: {threshold}",
        flush=True,
    )

    processed_sources = 0
    processed_pairs = 0
    matched_pairs = 0

    start_time = time.perf_counter()

    candidate_reader = pd.read_csv(
        candidate_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        chunksize=SOURCE_ROWS_PER_BATCH,
    )

    for batch_number, candidate_chunk in enumerate(
        candidate_reader,
        start=1,
    ):
        batch_start = time.perf_counter()

        pairs = expand_candidate_chunk(
            candidate_chunk
        )

        processed_sources += len(
            candidate_chunk
        )

        if pairs.empty:
            empty_results = pd.DataFrame(
                {
                    "source1_entity_id":
                        candidate_chunk[
                            "source1_entity_id"
                        ],
                    "matched_entity_ids": "",
                }
            )

            append_tsv(
                empty_results,
                result_path,
            )

            print(
                f"Batch {batch_number}: "
                f"{processed_sources:,} sources, "
                "0 candidate pairs",
                flush=True,
            )

            continue

        processed_pairs += len(pairs)

        print(
            f"\nBatch {batch_number}",
            flush=True,
        )

        print(
            f"Sources in batch: "
            f"{len(candidate_chunk):,}",
            flush=True,
        )

        print(
            f"Raw candidate pairs: "
            f"{len(pairs):,}",
            flush=True,
        )

        features = compute_pairwise_features(
            pairs,
            source_norm,
            target_norm,
            chunk_size=FEATURE_INNER_CHUNK,
            source_index=source_index,
            target_index=target_index,
        )

        if SAVE_FEATURES:
            append_tsv(
                features,
                feature_path,
            )

        scored = score_pairs(
            features,
            artifact=None,
        )

        scored["is_match"] = (
            scored["score"] >= threshold
        )

        if SAVE_SCORES:
            append_tsv(
                scored,
                score_path,
            )

        matches = scored.loc[
            scored["is_match"],
            PAIR_COLUMNS,
        ]

        matched_pairs += len(matches)

        match_map = {}

        for source_id, target_id in matches.itertuples(
            index=False,
            name=None,
        ):
            match_map.setdefault(
                source_id,
                set(),
            ).add(target_id)

        result_rows = []

        for source_id in candidate_chunk[
            "source1_entity_id"
        ]:
            targets = sorted(
                match_map.get(
                    source_id,
                    set(),
                )
            )

            result_rows.append(
                (
                    source_id,
                    ",".join(targets),
                )
            )

        results = pd.DataFrame(
            result_rows,
            columns=[
                "source1_entity_id",
                "matched_entity_ids",
            ],
        )

        append_tsv(
            results,
            result_path,
        )

        batch_seconds = (
            time.perf_counter()
            - batch_start
        )

        total_seconds = (
            time.perf_counter()
            - start_time
        )

        print(
            f"Completed batch {batch_number} "
            f"in {batch_seconds:.1f}s",
            flush=True,
        )

        print(
            f"Total sources processed: "
            f"{processed_sources:,}",
            flush=True,
        )

        print(
            f"Total candidate pairs processed: "
            f"{processed_pairs:,}",
            flush=True,
        )

        print(
            f"Total matches: "
            f"{matched_pairs:,}",
            flush=True,
        )

        print(
            f"Elapsed: "
            f"{total_seconds / 60:.1f} min",
            flush=True,
        )

        del pairs
        del features
        del scored
        del matches
        del results

    total_seconds = (
        time.perf_counter()
        - start_time
    )

    print("\nRESUME COMPLETE", flush=True)

    print(
        f"Sources processed: "
        f"{processed_sources:,}",
        flush=True,
    )

    print(
        f"Candidate pairs processed: "
        f"{processed_pairs:,}",
        flush=True,
    )

    print(
        f"Matches retained: "
        f"{matched_pairs:,}",
        flush=True,
    )

    print(
        f"Processing time: "
        f"{total_seconds / 60:.2f} minutes",
        flush=True,
    )

    print(
        f"Results: {result_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
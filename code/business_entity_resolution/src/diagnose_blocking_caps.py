from pathlib import Path
from collections import defaultdict

import pandas as pd

from blocking import blocking_keys
from config import PipelineConfig


SOURCE_PATH = Path(
    "output/baseline-aws-01/normalized_source1.tsv"
)

TARGET_PATH = Path(
    "output/baseline-aws-01/normalized_targets.tsv"
)

MISSES_PATH = Path(
    "output/baseline-aws-01/blocking_misses.tsv"
)

SAMPLE_SIZE = 200000
SEED = 42


def main():
    config = PipelineConfig()

    max_block_size = config.max_block_size
    max_candidates = config.max_candidates

    print(
        f"max_block_size = {max_block_size}",
        flush=True,
    )

    print(
        f"max_candidates = {max_candidates}",
        flush=True,
    )

    print("Loading misses...", flush=True)

    misses = pd.read_csv(
        MISSES_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    if len(misses) > SAMPLE_SIZE:
        misses = misses.sample(
            SAMPLE_SIZE,
            random_state=SEED,
        ).reset_index(drop=True)

    print(
        f"Sampled misses: {len(misses):,}",
        flush=True,
    )

    print("Loading normalized sources...", flush=True)

    source = pd.read_csv(
        SOURCE_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading normalized targets...", flush=True)

    target = pd.read_csv(
        TARGET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    source_index = pd.Index(
        source["entity_id"],
        copy=False,
    )

    target_index = pd.Index(
        target["entity_id"],
        copy=False,
    )

    source_positions = source_index.get_indexer(
        misses["source1_entity_id"]
    )

    target_positions = target_index.get_indexer(
        misses["target_entity_id"]
    )

    no_shared_key = 0
    country_conflict = 0

    eligible_per_miss = []

    interesting_blocks = set()

    print(
        "Finding shared eligible keys...",
        flush=True,
    )

    for i in range(len(misses)):
        sp = source_positions[i]
        tp = target_positions[i]

        if sp == -1 or tp == -1:
            eligible_per_miss.append([])
            continue

        s = source.iloc[sp]
        t = target.iloc[tp]

        source_keys = blocking_keys(s)
        target_keys = blocking_keys(t)

        common = source_keys & target_keys

        if not common:
            no_shared_key += 1
            eligible_per_miss.append([])
            continue

        source_country = s["norm_country"]
        target_country = t["norm_country"]

        if source_country:
            country_ok = (
                target_country == source_country
                or target_country == ""
            )
        else:
            country_ok = True

        if not country_ok:
            country_conflict += 1
            eligible_per_miss.append([])
            continue

        blocks = []

        for kind, key in common:
            block = (
                kind,
                key,
                target_country,
            )

            blocks.append(block)
            interesting_blocks.add(block)

        eligible_per_miss.append(blocks)

        if (i + 1) % 20000 == 0:
            print(
                f"Prepared {i + 1:,}/"
                f"{len(misses):,} misses",
                flush=True,
            )

    print(
        f"Blocks to inspect: "
        f"{len(interesting_blocks):,}",
        flush=True,
    )

    print(
        "Scanning targets for bucket sizes...",
        flush=True,
    )

    bucket_sizes = defaultdict(int)

    t_name = target[
        "norm_business_name"
    ].array

    t_address = target[
        "norm_business_address"
    ].array

    t_number = target[
        "street_number"
    ].array

    t_country = target[
        "norm_country"
    ].array

    for i in range(len(target)):
        record = {
            "norm_business_name":
                t_name[i],

            "norm_business_address":
                t_address[i],

            "street_number":
                t_number[i],

            "norm_country":
                t_country[i],
        }

        country = t_country[i]

        for kind, key in blocking_keys(
            record
        ):
            block = (
                kind,
                key,
                country,
            )

            if block not in interesting_blocks:
                continue

            # We only need to know whether the
            # bucket is <= limit or oversized.
            if bucket_sizes[block] <= max_block_size:
                bucket_sizes[block] += 1

        if (i + 1) % 500000 == 0:
            print(
                f"Scanned {i + 1:,}/"
                f"{len(target):,} targets",
                flush=True,
            )

    oversized_only = 0
    candidate_cap = 0
    unresolved = 0

    for blocks in eligible_per_miss:
        if not blocks:
            continue

        sizes = [
            bucket_sizes.get(
                block,
                0,
            )
            for block in blocks
        ]

        if not sizes or all(
            size == 0
            for size in sizes
        ):
            unresolved += 1
            continue

        # If even one shared eligible block is
        # small enough, the true target must have
        # entered the candidate pool. Since it is
        # in blocking_misses.tsv, it was then lost
        # at the max_candidates ranking cap.
        if any(
            0 < size <= max_block_size
            for size in sizes
        ):
            candidate_cap += 1

        else:
            oversized_only += 1

    classified = (
        no_shared_key
        + country_conflict
        + oversized_only
        + candidate_cap
        + unresolved
    )

    print()
    print("BLOCKING CAP DIAGNOSTICS")
    print("========================")

    categories = [
        (
            "No shared key",
            no_shared_key,
        ),
        (
            "Country conflict",
            country_conflict,
        ),
        (
            "Only oversized buckets",
            oversized_only,
        ),
        (
            "Lost at top-candidate cap",
            candidate_cap,
        ),
        (
            "Unresolved",
            unresolved,
        ),
    ]

    for name, count in categories:
        percentage = (
            count / classified
            if classified
            else 0.0
        )

        print(
            f"{name}: "
            f"{count:,} "
            f"({percentage:.2%})"
        )

    print()
    print(
        f"Classified sample: "
        f"{classified:,}"
    )


if __name__ == "__main__":
    main()
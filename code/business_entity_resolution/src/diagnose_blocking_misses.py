from pathlib import Path

import numpy as np
import pandas as pd

from blocking import blocking_keys


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


def clean(value):
    if isinstance(value, str):
        return value

    if pd.isna(value):
        return ""

    return str(value)


def main():
    print("Loading blocking misses...", flush=True)

    misses = pd.read_csv(
        MISSES_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        f"Total misses: {len(misses):,}",
        flush=True,
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

    print("Loading normalized source data...", flush=True)

    source = pd.read_csv(
        SOURCE_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Loading normalized target data...", flush=True)

    target = pd.read_csv(
        TARGET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print("Building entity indexes...", flush=True)

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
    shared_key = 0
    country_conflict = 0

    examples = []

    for i in range(len(misses)):
        sp = source_positions[i]
        tp = target_positions[i]

        if sp == -1 or tp == -1:
            continue

        s = source.iloc[sp]
        t = target.iloc[tp]

        source_keys = blocking_keys(s)
        target_keys = blocking_keys(t)

        common = source_keys & target_keys

        if not common:
            no_shared_key += 1

            if len(examples) < 20:
                examples.append(
                    {
                        "source1_entity_id":
                            s["entity_id"],

                        "target_entity_id":
                            t["entity_id"],

                        "source_name":
                            s["norm_business_name"],

                        "target_name":
                            t["norm_business_name"],

                        "source_address":
                            s["norm_business_address"],

                        "target_address":
                            t["norm_business_address"],

                        "reason":
                            "no_shared_key",
                    }
                )

            continue

        source_country = clean(
            s["norm_country"]
        )

        target_country = clean(
            t["norm_country"]
        )

        eligible = False

        if source_country:
            if (
                target_country == source_country
                or target_country == ""
            ):
                eligible = True
        else:
            eligible = True

        if eligible:
            shared_key += 1
        else:
            country_conflict += 1

            if len(examples) < 20:
                examples.append(
                    {
                        "source1_entity_id":
                            s["entity_id"],

                        "target_entity_id":
                            t["entity_id"],

                        "source_name":
                            s["norm_business_name"],

                        "target_name":
                            t["norm_business_name"],

                        "source_address":
                            s["norm_business_address"],

                        "target_address":
                            t["norm_business_address"],

                        "reason":
                            "country_scope_conflict",
                    }
                )

        if (
            (i + 1) % 10000
            == 0
        ):
            print(
                f"Processed {i + 1:,}/"
                f"{len(misses):,}",
                flush=True,
            )

    total = (
        no_shared_key
        + shared_key
        + country_conflict
    )

    print("\nBLOCKING MISS DIAGNOSTICS")
    print("=========================")

    print(
        f"No shared blocking key: "
        f"{no_shared_key:,} "
        f"({no_shared_key / total:.2%})"
    )

    print(
        f"Shared eligible key: "
        f"{shared_key:,} "
        f"({shared_key / total:.2%})"
    )

    print(
        f"Country scope conflict: "
        f"{country_conflict:,} "
        f"({country_conflict / total:.2%})"
    )

    pd.DataFrame(
        examples
    ).to_csv(
        "output/baseline-aws-01/"
        "blocking_miss_examples.tsv",
        sep="\t",
        index=False,
    )


if __name__ == "__main__":
    main()

from pathlib import Path
import json

import pandas as pd

from blocking_v2 import generate_candidate_file
from data_io import load_truth


SOURCE_PATH = Path(
    "output/baseline-aws-01/normalized_source1.tsv"
)

TARGET_PATH = Path(
    "output/baseline-aws-01/normalized_targets.tsv"
)

TRUTH_PATH = Path(
    "dataset/train_ground_truth.tsv"
)

OUTPUT_DIR = Path(
    "output/blocking-v2"
)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Loading normalized Source 1...",
        flush=True,
    )

    source = pd.read_csv(
        SOURCE_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        "Loading normalized targets...",
        flush=True,
    )

    targets = pd.read_csv(
        TARGET_PATH,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    print(
        "Loading truth...",
        flush=True,
    )

    truth = load_truth(
        TRUTH_PATH,
        source["entity_id"],
        targets["entity_id"],
    )

    diagnostics = generate_candidate_file(
        source,
        targets,
        OUTPUT_DIR
        / "candidate_pairs.tsv",
        truth=truth,
        max_candidates=50,
        max_block_size=500,
    )

    with open(
        OUTPUT_DIR
        / "blocking_report.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            diagnostics,
            handle,
            indent=2,
        )

    print()
    print("BLOCKING V2 COMPLETE")
    print("====================")

    for key, value in (
        diagnostics.items()
    ):
        print(
            f"{key}: {value}"
        )


if __name__ == "__main__":
    main()
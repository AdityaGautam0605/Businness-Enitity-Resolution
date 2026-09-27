import argparse
import csv
from pathlib import Path


CANDIDATES_PATH = Path(
    "output/baseline-aws-01/candidate_pairs.tsv"
)

SCORES_PATH = Path(
    "output/baseline-aws-resume-01/pair_scores.tsv"
)


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--threshold",
        type=float,
        required=True,
    )

    parser.add_argument(
        "--output",
        required=True,
    )

    args = parser.parse_args()

    threshold = args.threshold
    output_path = Path(args.output)

    if not 0 <= threshold <= 1:
        raise ValueError(
            "Threshold must be between 0 and 1"
        )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    processed_sources = 0
    processed_pairs = 0
    matched_pairs = 0

    print(
        f"Threshold: {threshold}",
        flush=True,
    )

    print(
        "Opening candidate and score files...",
        flush=True,
    )

    with open(
        CANDIDATES_PATH,
        encoding="utf-8-sig",
        newline="",
    ) as candidate_handle, open(
        SCORES_PATH,
        encoding="utf-8-sig",
        newline="",
    ) as score_handle, open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as output_handle:

        candidate_reader = csv.DictReader(
            candidate_handle,
            delimiter="\t",
        )

        score_reader = csv.DictReader(
            score_handle,
            delimiter="\t",
        )

        score_iterator = iter(score_reader)

        writer = csv.writer(
            output_handle,
            delimiter="\t",
            lineterminator="\n",
        )

        writer.writerow(
            [
                "source1_entity_id",
                "matched_entity_ids",
            ]
        )

        for candidate_row in candidate_reader:

            source_id = candidate_row[
                "source1_entity_id"
            ]

            packed = candidate_row[
                "candidate_entity_ids"
            ]

            if packed:
                candidate_ids = [
                    target.strip()
                    for target in packed.split(",")
                    if target.strip()
                ]
            else:
                candidate_ids = []

            expected_targets = set(
                candidate_ids
            )

            matches = set()

            for _ in range(
                len(candidate_ids)
            ):
                try:
                    score_row = next(
                        score_iterator
                    )
                except StopIteration:
                    raise RuntimeError(
                        "pair_scores.tsv ended "
                        "before candidate_pairs.tsv"
                    )

                score_source = score_row[
                    "source1_entity_id"
                ]

                target_id = score_row[
                    "target_entity_id"
                ]

                if score_source != source_id:
                    raise RuntimeError(
                        "Source ordering mismatch: "
                        f"expected {source_id}, "
                        f"found {score_source}"
                    )

                if target_id not in expected_targets:
                    raise RuntimeError(
                        "Target mismatch for "
                        f"{source_id}: "
                        f"{target_id}"
                    )

                score = float(
                    score_row["score"]
                )

                processed_pairs += 1

                if score >= threshold:
                    matches.add(
                        target_id
                    )

                    matched_pairs += 1

            writer.writerow(
                [
                    source_id,
                    ",".join(
                        sorted(matches)
                    ),
                ]
            )

            processed_sources += 1

            if (
                processed_sources
                % 100000
                == 0
            ):
                print(
                    f"Sources: "
                    f"{processed_sources:,} | "
                    f"Pairs: "
                    f"{processed_pairs:,} | "
                    f"Matches: "
                    f"{matched_pairs:,}",
                    flush=True,
                )

        # Make sure pair_scores.tsv did not contain
        # unexpected extra rows.
        try:
            extra = next(
                score_iterator
            )

            raise RuntimeError(
                "pair_scores.tsv contains "
                "unexpected extra rows "
                f"starting with "
                f"{extra['source1_entity_id']}"
            )

        except StopIteration:
            pass

    print(
        "\nFINAL GENERATION COMPLETE",
        flush=True,
    )

    print(
        f"Sources: "
        f"{processed_sources:,}",
        flush=True,
    )

    print(
        f"Candidate pairs: "
        f"{processed_pairs:,}",
        flush=True,
    )

    print(
        f"Matches: "
        f"{matched_pairs:,}",
        flush=True,
    )

    print(
        f"Output: {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
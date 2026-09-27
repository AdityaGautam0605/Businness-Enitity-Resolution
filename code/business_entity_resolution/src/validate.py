"""Backend: strict submission format and referential integrity validation."""
import argparse
from data_io import load_sources, read_id_lists


def validate_outputs(matching_file, candidate_file, s1_file, s2_file, s3_file):
    try:
        source, targets, _ = load_sources(s1_file, s2_file, s3_file)
        matches = read_id_lists(matching_file, "matched_entity_ids")
        candidates = read_id_lists(candidate_file, "candidate_entity_ids")
        source_ids, target_ids = set(source.entity_id), set(targets.entity_id)
        if set(matches) != source_ids or set(candidates) != source_ids:
            raise ValueError("Both outputs must cover every Source 1 ID exactly once")
        for source_id in source_ids:
            if candidates[source_id] - target_ids or matches[source_id] - target_ids:
                raise ValueError(f"Unknown target ID for {source_id}")
            if matches[source_id] - candidates[source_id]:
                raise ValueError(f"Match was not a candidate for {source_id}")
    except (ValueError, OSError) as exc:
        print(f"[FAIL] {exc}")
        return False
    print("[PASS] Output schemas, coverage, IDs and candidate membership are valid.")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("matching", "candidates", "s1", "s2", "s3"):
        parser.add_argument(name)
    args = parser.parse_args()
    raise SystemExit(0 if validate_outputs(args.matching, args.candidates, args.s1, args.s2, args.s3) else 1)

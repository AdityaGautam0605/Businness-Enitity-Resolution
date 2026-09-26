import sys
import pandas as pd

def validate_outputs(matching_file: str, candidate_file: str, s1_file: str, s2_file: str, s3_file: str) -> bool:
    print("=" * 60)
    print("RUNNING SUBMISSION VALIDATION CHECKS")
    print("=" * 60)

    s1_ids = set(pd.read_csv(s1_file, sep="\t", dtype=str)["entity_id"].dropna())
    valid_target_ids = set(pd.read_csv(s2_file, sep="\t", dtype=str)["entity_id"].dropna()) | \
                       set(pd.read_csv(s3_file, sep="\t", dtype=str)["entity_id"].dropna())

    try:
        match_df = pd.read_csv(matching_file, sep="\t", dtype=str, keep_default_na=False)
        cand_df = pd.read_csv(candidate_file, sep="\t", dtype=str, keep_default_na=False)
    except Exception as e:
        print(f"[FAIL] TSV read error: {e}")
        return False

    if list(match_df.columns) != ["source1_entity_id", "matched_entity_ids"]:
        print(f"[FAIL] matching_results.tsv headers mismatch: {list(match_df.columns)}")
        return False

    if list(cand_df.columns) != ["source1_entity_id", "candidate_entity_ids"]:
        print(f"[FAIL] candidate_pairs.tsv headers mismatch: {list(cand_df.columns)}")
        return False

    match_s1 = set(match_df["source1_entity_id"])
    cand_s1 = set(cand_df["source1_entity_id"])

    if match_s1 != s1_ids:
        print(f"[FAIL] matching_results.tsv lacks 1-to-1 Source 1 coverage! Missing: {len(s1_ids - match_s1)}")
        return False

    if cand_s1 != s1_ids:
        print(f"[FAIL] candidate_pairs.tsv lacks 1-to-1 Source 1 coverage! Missing: {len(s1_ids - cand_s1)}")
        return False

    if len(match_df) != len(s1_ids) or len(cand_df) != len(s1_ids):
        print("[FAIL] Duplicate rows detected.")
        return False

    cand_dict = {}
    for _, row in cand_df.iterrows():
        s1 = row["source1_entity_id"]
        cands = [c.strip() for c in str(row["candidate_entity_ids"]).split(",") if c.strip()]
        if len(cands) != len(set(cands)):
            print(f"[FAIL] Duplicate candidate IDs detected for {s1}")
            return False
        for c in cands:
            if c not in valid_target_ids:
                print(f"[FAIL] Illegal candidate ID '{c}'")
                return False
        cand_dict[s1] = set(cands)

    for _, row in match_df.iterrows():
        s1 = row["source1_entity_id"]
        matches = [m.strip() for m in str(row["matched_entity_ids"]).split(",") if m.strip()]
        if len(matches) != len(set(matches)):
            print(f"[FAIL] Duplicate match IDs detected for {s1}")
            return False
        for m in matches:
            if m not in valid_target_ids:
                print(f"[FAIL] Illegal matched ID '{m}'")
                return False
            if m not in cand_dict.get(s1, set()):
                print(f"[FAIL] Matched ID '{m}' was not in candidate_pairs for entity {s1}")
                return False

    print("[PASS] Validation passed. Both files are compliant with the requirements.")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 6:
        print("Usage: python src/validate.py <matching.tsv> <candidate.tsv> <s1.tsv> <s2.tsv> <s3.tsv>")
        sys.exit(1)
    passed = validate_outputs(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5])
    sys.exit(0 if passed else 1)
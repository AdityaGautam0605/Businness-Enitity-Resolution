import os
import sys
import time
import argparse
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

from normalize import normalize_dataset
from blocking import generate_candidate_pairs
from features import compute_pairwise_features
from model import predict_matches
from aggregate import aggregate_to_tsv_format
from validate import validate_outputs

def run(s1_path: str, s2_path: str, s3_path: str, out_dir: str):
    total_start = time.time()
    os.makedirs(out_dir, exist_ok=True)
    match_file = os.path.join(out_dir, "matching_results.tsv")
    cand_file = os.path.join(out_dir, "candidate_pairs.tsv")

    print("[1/5] Ingesting & Normalizing Sources...")
    s1_df = pd.read_csv(s1_path, sep="\t", dtype=str)
    s2_df = pd.read_csv(s2_path, sep="\t", dtype=str)
    s3_df = pd.read_csv(s3_path, sep="\t", dtype=str)

    target_combined = pd.concat([s2_df, s3_df], ignore_index=True)

    s1_norm = normalize_dataset(s1_df)
    target_norm = normalize_dataset(target_combined)

    print("[2/5] Running Blocking / Candidate Generation...")
    candidate_pairs = generate_candidate_pairs(s1_norm, target_norm)
    all_s1_ids = s1_df["entity_id"].dropna().unique()

    cand_aggregated = aggregate_to_tsv_format(all_s1_ids, candidate_pairs, "candidate_entity_ids")
    cand_aggregated.to_csv(cand_file, sep="\t", index=False)
    print(f"      Saved {cand_file} ({len(candidate_pairs)} candidate pairs shortlisted)")

    print("[3/5] Extracting Pairwise Features...")
    features = compute_pairwise_features(candidate_pairs, s1_norm, target_norm)

    print("[4/5] ML Matching Inference & Precision Thresholding...")
    matches_pairwise = predict_matches(features)

    print("[5/5] Aggregating Final Matches...")
    matches_aggregated = aggregate_to_tsv_format(all_s1_ids, matches_pairwise, "matched_entity_ids")
    matches_aggregated.to_csv(match_file, sep="\t", index=False)
    print(f"      Saved {match_file}")

    print(f"[*] Pipeline finished in {time.time() - total_start:.2f}s")

    if not validate_outputs(match_file, cand_file, s1_path, s2_path, s3_path):
        print("[CRITICAL] Pipeline outputs failed validation!")
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--s1", required=True)
    parser.add_argument("--s2", required=True)
    parser.add_argument("--s3", required=True)
    parser.add_argument("--out", default="output")
    args = parser.parse_args()

    run(args.s1, args.s2, args.s3, args.out)
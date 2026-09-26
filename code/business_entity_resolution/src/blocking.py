import pandas as pd
import re

def extract_clean_stem(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip common business prefixes/noise words
    noise = r"\b(the|shree|shri|dr|om|m/s|a|an|new)\b"
    cleaned = re.sub(noise, "", text.lower()).strip()
    # Extract only alphanumeric characters
    cleaned = re.sub(r"[^\w]", "", cleaned)
    return cleaned[:4] if len(cleaned) >= 3 else ""

def generate_candidate_pairs(s1_df: pd.DataFrame, target_df: pd.DataFrame, max_cands_per_s1: int = 15) -> pd.DataFrame:
    print("      Building composite blocking index...")
    s1 = s1_df[["entity_id", "norm_business_name", "norm_country"]].copy()
    tgt = target_df[["entity_id", "norm_business_name", "norm_country"]].copy()

    s1["stem"] = s1["norm_business_name"].apply(extract_clean_stem)
    tgt["stem"] = tgt["norm_business_name"].apply(extract_clean_stem)

    # Discard records with empty stems
    s1 = s1[s1["stem"] != ""]
    tgt = tgt[tgt["stem"] != ""]

    # Block Key: Country + Clean 4-char Stem
    s1["block_key"] = s1["norm_country"] + "::" + s1["stem"]
    tgt["block_key"] = tgt["norm_country"] + "::" + tgt["stem"]

    # Discard overly dense keys (stop oversized buckets)
    counts = tgt["block_key"].value_counts()
    safe_keys = set(counts[counts <= 300].index)

    s1 = s1[s1["block_key"].isin(safe_keys)]
    tgt = tgt[tgt["block_key"].isin(safe_keys)]

    print(f"      Merging on {len(safe_keys)} clean keys...")
    pairs = pd.merge(
        s1[["entity_id", "block_key"]].rename(columns={"entity_id": "source1_entity_id"}),
        tgt[["entity_id", "block_key"]].rename(columns={"entity_id": "target_entity_id"}),
        on="block_key"
    )[["source1_entity_id", "target_entity_id"]].drop_duplicates()

    # Cap to max 15 candidates per S1 entity
    if not pairs.empty:
        pairs = pairs.groupby("source1_entity_id").head(max_cands_per_s1).reset_index(drop=True)

    print(f"      Shortlisted {len(pairs)} candidate pairs.")
    return pairs
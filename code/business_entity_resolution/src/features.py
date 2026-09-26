import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

def compute_pairwise_features(pairs_df: pd.DataFrame, s1_df: pd.DataFrame, tgt_df: pd.DataFrame) -> pd.DataFrame:
    if pairs_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "target_entity_id", "token_sort", "address_ratio"])

    name_map_s1 = dict(zip(s1_df["entity_id"], s1_df["norm_business_name"].fillna("")))
    addr_map_s1 = dict(zip(s1_df["entity_id"], s1_df["norm_business_address"].fillna("")))

    name_map_tgt = dict(zip(tgt_df["entity_id"], tgt_df["norm_business_name"].fillna("")))
    addr_map_tgt = dict(zip(tgt_df["entity_id"], tgt_df["norm_business_address"].fillna("")))

    s1_ids = pairs_df["source1_entity_id"].tolist()
    tgt_ids = pairs_df["target_entity_id"].tolist()

    token_sorts = []
    addr_ratios = []

    print(f"      Scoring {len(pairs_df)} candidate pairs...")
    for s1_id, tgt_id in zip(s1_ids, tgt_ids):
        n1 = name_map_s1.get(s1_id, "")
        n2 = name_map_tgt.get(tgt_id, "")
        a1 = addr_map_s1.get(s1_id, "")
        a2 = addr_map_tgt.get(tgt_id, "")

        # Fast string metric
        token_sorts.append(fuzz.token_sort_ratio(n1, n2) / 100.0)
        addr_ratios.append(fuzz.ratio(a1, a2) / 100.0)

    features_df = pd.DataFrame({
        "source1_entity_id": s1_ids,
        "target_entity_id": tgt_ids,
        "token_sort": token_sorts,
        "address_ratio": addr_ratios
    })

    return features_df
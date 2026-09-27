"""ML-2: ordered, versioned feature contract shared with persisted models."""
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from blocking import PAIR_COLUMNS

FEATURE_VERSION = 1
FEATURE_COLUMNS = ["token_sort", "name_token_set", "name_jaro_winkler", "name_ratio", "name_jaccard", "address_ratio", "address_token_sort", "address_jaccard", "street_number_match", "street_number_conflict", "city_match", "state_match", "postal_code_match", "country_match", "name_present", "address_present"]


def jaccard(a, b):
    left, right = set(a.split()), set(b.split())
    return len(left & right) / len(left | right) if left and right else 0.0


def exact(a, b):
    return float(bool(a and b) and a == b)


def compute_pairwise_features(pairs_df, s1_df, tgt_df):
    if pairs_df.empty:
        return pd.DataFrame(columns=PAIR_COLUMNS + FEATURE_COLUMNS)
    if s1_df.entity_id.duplicated().any() or tgt_df.entity_id.duplicated().any():
        raise ValueError("Feature lookup requires unique entity IDs")
    sources = s1_df.set_index("entity_id").to_dict("index")
    targets = tgt_df.set_index("entity_id").to_dict("index")
    rows = []
    for source_id, target_id in pairs_df[PAIR_COLUMNS].itertuples(index=False, name=None):
        if source_id not in sources or target_id not in targets:
            raise ValueError(f"Candidate references unknown entity: {source_id}, {target_id}")
        s, t = sources[source_id], targets[target_id]
        n1, n2 = s["norm_business_name"], t["norm_business_name"]
        a1, a2 = s["norm_business_address"], t["norm_business_address"]
        names, addresses = bool(n1 and n2), bool(a1 and a2)
        num1, num2 = s["street_number"], t["street_number"]
        rows.append([source_id, target_id,
            fuzz.token_sort_ratio(n1, n2) / 100 if names else 0.0,
            fuzz.token_set_ratio(n1, n2) / 100 if names else 0.0,
            JaroWinkler.normalized_similarity(n1, n2) if names else 0.0,
            fuzz.ratio(n1, n2) / 100 if names else 0.0,
            jaccard(n1, n2),
            fuzz.ratio(a1, a2) / 100 if addresses else 0.0,
            fuzz.token_sort_ratio(a1, a2) / 100 if addresses else 0.0,
            jaccard(a1, a2), exact(num1, num2), float(bool(num1 and num2) and num1 != num2),
            *[exact(s[f"norm_{col}"], t[f"norm_{col}"]) for col in ("city", "state", "postal_code", "country")],
            float(names), float(addresses)])
    return pd.DataFrame(rows, columns=PAIR_COLUMNS + FEATURE_COLUMNS)

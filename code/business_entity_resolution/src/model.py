import pandas as pd

def predict_matches(feature_df: pd.DataFrame, threshold: float = 0.86) -> pd.DataFrame:
    """
    ML-3 classifier / baseline decision threshold.
    Calibrated with precision bias for F0.5.
    """
    if feature_df.empty:
        return pd.DataFrame(columns=["source1_entity_id", "target_entity_id"])

    feature_df["score"] = (
        0.65 * feature_df["token_sort"] + 
        0.35 * feature_df["address_ratio"]
    )
    is_match = feature_df["score"] >= threshold
    return feature_df[is_match][["source1_entity_id", "target_entity_id"]]
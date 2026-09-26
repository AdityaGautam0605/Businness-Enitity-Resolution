import pandas as pd

def aggregate_to_tsv_format(all_s1_ids: list, pairwise_df: pd.DataFrame, target_col_name: str) -> pd.DataFrame:
    """
    Collapses pairwise records into one row per Source 1 ID.
    Singletons receive empty strings.
    """
    base_df = pd.DataFrame({"source1_entity_id": list(all_s1_ids)})

    if pairwise_df.empty:
        base_df[target_col_name] = ""
        return base_df

    grouped = (
        pairwise_df.groupby("source1_entity_id")["target_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(str(x) for x in ids if str(x).strip()))))
        .reset_index()
        .rename(columns={"target_entity_id": target_col_name})
    )

    result = pd.merge(base_df, grouped, on="source1_entity_id", how="left")
    result[target_col_name] = result[target_col_name].fillna("")
    return result[["source1_entity_id", target_col_name]]
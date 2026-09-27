"""Collapse pair decisions to the submission contract."""
import pandas as pd
from blocking import PAIR_COLUMNS


def aggregate_to_tsv_format(all_s1_ids, pairwise_df, target_col_name):
    ids = list(all_s1_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Aggregation requires unique Source 1 IDs")
    if set(pairwise_df.source1_entity_id) - set(ids):
        raise ValueError("Pairwise data contains unknown Source 1 IDs")
    grouped = {}
    for source, target in pairwise_df[PAIR_COLUMNS].itertuples(index=False, name=None):
        if not isinstance(target, str) or not target or target != target.strip() or "," in target:
            raise ValueError("Invalid target ID during aggregation")
        grouped.setdefault(source, set()).add(target)
    return pd.DataFrame({
        "source1_entity_id": ids,
        target_col_name: [",".join(sorted(grouped.get(source, set()))) for source in ids],
    })

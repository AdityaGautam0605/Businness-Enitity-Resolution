"""Strict input contracts and JSON diagnostics; IDs stay strings."""
import csv
import json
from pathlib import Path
import pandas as pd

TEXT_COLUMNS = ("business_name", "business_address", "country", "city", "state", "postal_code")


def read_tsv(path):
    with open(path, encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter="\t"), [])
    if not header or any(not col or col != col.strip() for col in header) or len(set(header)) != len(header):
        raise ValueError(f"{path}: headers must be unique, nonempty and trimmed")
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False, encoding="utf-8-sig", skip_blank_lines=False)
    if not isinstance(df.index, pd.RangeIndex):
        raise ValueError(f"{path}: row has more fields than the header; check tab separators")
    return df


def read_source(path):
    df = read_tsv(path)
    if "entity_id" not in df:
        raise ValueError(f"{path}: required column 'entity_id' is missing")
    ids = df["entity_id"]
    if ids.str.strip().eq("").any() or ids.ne(ids.str.strip()).any():
        raise ValueError(f"{path}: entity IDs must be nonempty and have no surrounding whitespace")
    if ids.str.contains(r"[,\r\n\t]", regex=True).any():
        raise ValueError(f"{path}: entity IDs cannot contain commas or control separators")
    if ids.duplicated().any():
        raise ValueError(f"{path}: duplicate entity IDs: {ids[ids.duplicated()].head().tolist()}")
    for col in TEXT_COLUMNS:
        if col not in df:
            df[col] = ""
    return df


def load_sources(s1_path, s2_path, s3_path):
    s1, s2, s3 = [read_source(p) for p in (s1_path, s2_path, s3_path)]
    overlap = set(s2.entity_id) & set(s3.entity_id)
    if overlap:
        raise ValueError(f"Sources 2 and 3 must have globally unique target IDs; collisions: {sorted(overlap)[:5]}")
    return s1, pd.concat([s2, s3], ignore_index=True), {
        "source1": profile(s1), "source2": profile(s2), "source3": profile(s3)
    }


def profile(df):
    return {
        "records": len(df),
        "missing_fields": {c: int(df[c].str.strip().eq("").sum()) for c in TEXT_COLUMNS},
        "duplicate_name_address_rows": int(df.duplicated(["business_name", "business_address"]).sum()),
        "countries": df.country.value_counts().to_dict(),
    }


def read_id_lists(path, value_column):
    df = read_tsv(path)
    if list(df.columns) != ["source1_entity_id", value_column]:
        raise ValueError(f"{path}: expected columns source1_entity_id, {value_column}")
    ids = df.source1_entity_id
    if ids.duplicated().any() or ids.str.strip().eq("").any() or ids.ne(ids.str.strip()).any():
        raise ValueError(f"{path}: source IDs must be unique, nonempty and trimmed")
    result = {}
    for source_id, raw in df.itertuples(index=False, name=None):
        values = raw.split(",") if raw else []
        if any(not v or v != v.strip() for v in values) or len(set(values)) != len(values):
            raise ValueError(f"{path}: malformed or duplicate target IDs for {source_id}")
        result[source_id] = set(values)
    return result


def load_truth(path, source_ids, target_ids):
    truth = read_id_lists(path, "matched_entity_ids")
    unknown_sources = set(truth) - set(source_ids)
    unknown_targets = set().union(*truth.values()) - set(target_ids) if truth else set()
    if unknown_sources or unknown_targets:
        raise ValueError(f"Ground truth contains unknown IDs: sources={sorted(unknown_sources)[:5]}, targets={sorted(unknown_targets)[:5]}")
    return truth


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")

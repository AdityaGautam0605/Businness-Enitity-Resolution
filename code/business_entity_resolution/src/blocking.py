"""ML-1: union multiple indexed passes, then rank candidates deterministically."""
from collections import defaultdict
import re
import pandas as pd
from rapidfuzz import fuzz

PAIR_COLUMNS = ["source1_entity_id", "target_entity_id"]


def soundex(text):
    letters = re.sub("[^a-z]", "", text.lower())
    if not letters:
        return ""
    codes = {c: str(i) for i, chars in enumerate(("bfpv", "cgjkqsxz", "dt", "l", "mn", "r"), 1) for c in chars}
    result, previous = letters[0], codes.get(letters[0], "")
    for char in letters[1:]:
        code = codes.get(char, "")
        if code and code != previous:
            result += code
        if char not in "hw":
            previous = code
    return (result + "000")[:4]


def extract_clean_stem(text):
    return re.sub(r"\W", "", text)[:4] if isinstance(text, str) else ""


def blocking_keys(record):
    name, address = record["norm_business_name"], record["norm_business_address"]
    tokens = name.split()
    keys = set()
    stem = extract_clean_stem(name)
    if len(stem) >= 3:
        keys.add(("prefix", stem))
    for token in tokens:
        if len(token) >= 3:
            keys.add(("name_token", token))
    if tokens:
        phonetic = soundex(tokens[0])
        if phonetic:
            keys.add(("phonetic", phonetic))
    if address:
        keys.add(("address", address))
    number = record["street_number"]
    street = [t for t in address.split() if not any(c.isdigit() for c in t) and t not in {"street", "road", "avenue", "lane", "suite", "apartment"}]
    if number and street:
        keys.add(("street", number + ":" + street[0][:5]))
    return keys


def generate_candidate_pairs(s1_df, target_df, max_cands_per_s1=50, max_block_size=500, diagnostics=None):
    if max_cands_per_s1 < 1 or max_block_size < 1:
        raise ValueError("Candidate and block limits must be positive")
    records = target_df.to_dict("records")
    index = defaultdict(list)
    for i, record in enumerate(records):
        for kind, key in blocking_keys(record):
            index[(kind, key, record["norm_country"])].append(i)
    countries = defaultdict(set)
    for kind, key, country in index:
        countries[(kind, key)].add(country)
    skipped, counts, pairs = 0, [], []
    for source in s1_df.to_dict("records"):
        candidates = set()
        country = source["norm_country"]
        for kind, key in sorted(blocking_keys(source)):
            scopes = {country, ""} if country else countries.get((kind, key), set())
            for scope in sorted(scopes):
                bucket = index.get((kind, key, scope), [])
                if len(bucket) > max_block_size:
                    skipped += 1
                    continue
                candidates.update(bucket)
        def rank(i):
            target = records[i]
            name_score = fuzz.token_sort_ratio(source["norm_business_name"], target["norm_business_name"]) if source["norm_business_name"] and target["norm_business_name"] else 0
            address_score = fuzz.token_sort_ratio(source["norm_business_address"], target["norm_business_address"]) if source["norm_business_address"] and target["norm_business_address"] else 0
            return (-0.7 * name_score - 0.3 * address_score, target["entity_id"])
        counts.append(len(candidates))
        for i in sorted(candidates, key=rank)[:max_cands_per_s1]:
            pairs.append((source["entity_id"], records[i]["entity_id"]))
    if diagnostics is not None:
        diagnostics.update({"pairs": len(pairs), "sources_without_candidates": sum(c == 0 for c in counts), "sources_capped": sum(c > max_cands_per_s1 for c in counts), "oversized_bucket_queries_skipped": skipped, "max_candidates_before_cap": max(counts, default=0)})
    return pd.DataFrame(pairs, columns=PAIR_COLUMNS)


def blocking_recall(pairs, truth):
    predicted = defaultdict(set)
    for source, target in pairs[PAIR_COLUMNS].itertuples(index=False, name=None):
        predicted[source].add(target)
    total = sum(len(ids) for ids in truth.values())
    found = sum(len(ids & predicted[source]) for source, ids in truth.items())
    return {"true_pairs": total, "retained_true_pairs": found, "blocking_recall": found / total if total else None}

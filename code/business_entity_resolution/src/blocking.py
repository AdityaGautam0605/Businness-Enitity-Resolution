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


def generate_candidate_pairs(
    s1_df,
    target_df,
    max_cands_per_s1=50,
    max_block_size=500,
    diagnostics=None,
):
    import heapq

    if max_cands_per_s1 < 1 or max_block_size < 1:
        raise ValueError("Candidate and block limits must be positive")

    # ---------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------

    def as_text(value):
        if isinstance(value, str):
            return value

        if pd.isna(value):
            return ""

        return str(value)

    def keys_from_values(name, address, number):
        name = as_text(name)
        address = as_text(address)
        number = as_text(number)

        tokens = name.split()
        keys = set()

        # Business-name prefix
        stem = extract_clean_stem(name)

        if len(stem) >= 3:
            keys.add(("prefix", stem))

        # Individual name tokens
        for token in tokens:
            if len(token) >= 3:
                keys.add(("name_token", token))

        # Soundex of first token
        if tokens:
            phonetic = soundex(tokens[0])

            if phonetic:
                keys.add(("phonetic", phonetic))

        # Full normalized address
        if address:
            keys.add(("address", address))

        # Street-number + first useful street token
        street = [
            token
            for token in address.split()
            if not any(c.isdigit() for c in token)
            and token
            not in {
                "street",
                "road",
                "avenue",
                "lane",
                "suite",
                "apartment",
            }
        ]

        if number and street:
            keys.add(
                (
                    "street",
                    number + ":" + street[0][:5],
                )
            )

        return keys

    # =========================================================
    # TARGET COLUMNS
    #
    # IMPORTANT:
    # These are references to DataFrame columns.
    #
    # We DO NOT do:
    #
    # target_df.to_dict("records")
    #
    # or:
    #
    # list(target_df.itertuples(...))
    #
    # Both create massive second copies of the data.
    # =========================================================

    t_entity = target_df["entity_id"].array
    t_name = target_df["norm_business_name"].array
    t_address = target_df["norm_business_address"].array
    t_number = target_df["street_number"].array
    t_country = target_df["norm_country"].array

    # =========================================================
    # BUILD BLOCKING INDEX
    # =========================================================

    index = defaultdict(list)

    # Used when source country is unknown.
    countries = defaultdict(set)

    for i in range(len(target_df)):
        name = t_name[i]
        address = t_address[i]
        number = t_number[i]
        country = as_text(t_country[i])

        keys = keys_from_values(
            name,
            address,
            number,
        )

        for kind, key in keys:
            block_key = (
                kind,
                key,
                country,
            )

            bucket = index[block_key]

            # -------------------------------------------------
            # MEMORY OPTIMIZATION
            #
            # Original code stored EVERY matching target:
            #
            # [1, 5, 19, 50, ... potentially millions ...]
            #
            # But the algorithm later ignores buckets larger
            # than max_block_size anyway.
            #
            # Therefore we only need:
            #
            # max_block_size + 1
            #
            # entries.
            #
            # The extra element tells us:
            #
            # "this bucket is oversized"
            # -------------------------------------------------

            if len(bucket) <= max_block_size:
                bucket.append(i)

            countries[(kind, key)].add(country)

    # =========================================================
    # SOURCE COLUMN REFERENCES
    # =========================================================

    s_entity = s1_df["entity_id"].array
    s_name = s1_df["norm_business_name"].array
    s_address = s1_df["norm_business_address"].array
    s_number = s1_df["street_number"].array
    s_country = s1_df["norm_country"].array

    skipped = 0

    pairs = []

    sources_without_candidates = 0
    sources_capped = 0
    max_candidates_before_cap = 0

    # =========================================================
    # PROCESS SOURCES ONE AT A TIME
    # =========================================================

    for source_idx in range(len(s1_df)):

        source_name = as_text(
            s_name[source_idx]
        )

        source_address = as_text(
            s_address[source_idx]
        )

        source_number = s_number[source_idx]

        country = as_text(
            s_country[source_idx]
        )

        candidates = set()

        source_keys = keys_from_values(
            source_name,
            source_address,
            source_number,
        )

        # Deterministic processing
        for kind, key in sorted(source_keys):

            # If country is known:
            #
            # exact country + targets with missing country
            #
            if country:
                scopes = (
                    country,
                    "",
                )

            # Otherwise search all countries where this key exists.
            else:
                scopes = countries.get(
                    (kind, key),
                    (),
                )

            for scope in sorted(scopes):

                bucket = index.get(
                    (
                        kind,
                        key,
                        scope,
                    )
                )

                if not bucket:
                    continue

                # Bucket was deliberately truncated because it
                # exceeded max_block_size.
                if len(bucket) > max_block_size:
                    skipped += 1
                    continue

                candidates.update(bucket)

        # =====================================================
        # DIAGNOSTICS
        # =====================================================

        num_candidates = len(candidates)

        if num_candidates == 0:
            sources_without_candidates += 1

        if num_candidates > max_cands_per_s1:
            sources_capped += 1

        if num_candidates > max_candidates_before_cap:
            max_candidates_before_cap = num_candidates

        # =====================================================
        # RANK CANDIDATES
        # =====================================================

        def rank(i):
            target_name = as_text(
                t_name[i]
            )

            target_address = as_text(
                t_address[i]
            )

            if source_name and target_name:
                name_score = fuzz.token_sort_ratio(
                    source_name,
                    target_name,
                )
            else:
                name_score = 0

            if source_address and target_address:
                address_score = fuzz.token_sort_ratio(
                    source_address,
                    target_address,
                )
            else:
                address_score = 0

            score = (
                0.7 * name_score
                + 0.3 * address_score
            )

            # Negative because heapq.nsmallest()
            # should return highest similarity first.
            return (
                -score,
                str(t_entity[i]),
            )

        # -----------------------------------------------------
        # Instead of:
        #
        # sorted(candidates, key=rank)[:50]
        #
        # which sorts ALL candidates,
        #
        # nsmallest only keeps the best N.
        # -----------------------------------------------------

        best_candidates = heapq.nsmallest(
            max_cands_per_s1,
            candidates,
            key=rank,
        )

        source_entity_id = s_entity[source_idx]

        for target_idx in best_candidates:
            pairs.append(
                (
                    source_entity_id,
                    t_entity[target_idx],
                )
            )

    # =========================================================
    # DIAGNOSTICS OUTPUT
    # =========================================================

    if diagnostics is not None:
        diagnostics.update(
            {
                "pairs": len(pairs),

                "sources_without_candidates":
                    sources_without_candidates,

                "sources_capped":
                    sources_capped,

                "oversized_bucket_queries_skipped":
                    skipped,

                "max_candidates_before_cap":
                    max_candidates_before_cap,
            }
        )

    return pd.DataFrame(
        pairs,
        columns=PAIR_COLUMNS,
    )
def blocking_recall(pairs, truth):
    predicted = defaultdict(set)
    for source, target in pairs[PAIR_COLUMNS].itertuples(index=False, name=None):
        predicted[source].add(target)
    total = sum(len(ids) for ids in truth.values())
    found = sum(len(ids & predicted[source]) for source, ids in truth.items())
    return {"true_pairs": total, "retained_true_pairs": found, "blocking_recall": found / total if total else None}

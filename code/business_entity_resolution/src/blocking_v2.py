from array import array
from collections import defaultdict
import heapq
import re

import pandas as pd
from rapidfuzz import fuzz

from blocking import (
    PAIR_COLUMNS,
    extract_clean_stem,
    soundex,
)


GENERIC_ADDRESS_TOKENS = {
    "street",
    "road",
    "avenue",
    "lane",
    "suite",
    "apartment",
    "building",
    "floor",
}


def as_text(value):
    if isinstance(value, str):
        return value

    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    return str(value)


def base_keys(name, address, number):
    name = as_text(name)
    address = as_text(address)
    number = as_text(number)

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

    street_tokens = [
        token
        for token in address.split()
        if len(token) >= 3
        and not any(
            character.isdigit()
            for character in token
        )
        and token not in GENERIC_ADDRESS_TOKENS
    ]

    if number and street_tokens:
        keys.add(
            (
                "street",
                number + ":" + street_tokens[0][:5],
            )
        )

    return keys


def char_trigrams(text):
    compact = re.sub(
        r"[^a-z0-9]",
        "",
        as_text(text).lower(),
    )

    if len(compact) < 3:
        return set()

    return {
        compact[i:i + 3]
        for i in range(len(compact) - 2)
    }


def refinement_keys(
    name,
    address,
    number,
    city,
    state,
    postal,
):
    name = as_text(name)
    address = as_text(address)
    number = as_text(number)
    city = as_text(city)
    state = as_text(state)
    postal = as_text(postal)

    result = set()

    name_tokens = [
        token
        for token in name.split()
        if len(token) >= 3
    ]

    address_tokens = [
        token
        for token in address.split()
        if len(token) >= 3
        and token not in GENERIC_ADDRESS_TOKENS
        and not token.isdigit()
    ]

    stem = extract_clean_stem(name)

    # Strong geographic keys.
    if postal:
        result.add(
            ("postal", postal)
        )

    if city and stem:
        result.add(
            (
                "city_prefix",
                city + ":" + stem,
            )
        )

    if state and stem:
        result.add(
            (
                "state_prefix",
                state + ":" + stem,
            )
        )

    if city and number:
        result.add(
            (
                "city_number",
                city + ":" + number,
            )
        )

    if postal and stem:
        result.add(
            (
                "postal_prefix",
                postal + ":" + stem,
            )
        )

    # Multiple name tokens, order independent.
    unique_name_tokens = sorted(
        set(name_tokens)
    )

    if len(unique_name_tokens) >= 2:
        for i in range(
            min(len(unique_name_tokens), 4)
        ):
            for j in range(
                i + 1,
                min(len(unique_name_tokens), 4),
            ):
                result.add(
                    (
                        "name_pair",
                        unique_name_tokens[i]
                        + ":"
                        + unique_name_tokens[j],
                    )
                )

    # Any useful additional name token.
    for token in unique_name_tokens[:6]:
        result.add(
            ("ref_name_token", token)
        )

    # Address information.
    if number:
        result.add(
            ("street_number", number)
        )

    for token in sorted(
        set(address_tokens)
    )[:5]:
        result.add(
            ("address_token", token)
        )

        if number:
            result.add(
                (
                    "number_address_token",
                    number + ":" + token,
                )
            )

    # Character-level fallback for noisy names.
    for trigram in sorted(
        char_trigrams(name)
    )[:10]:
        result.add(
            ("name_trigram", trigram)
        )

    return result


def generate_candidate_file(
    source_df,
    target_df,
    output_path,
    truth=None,
    max_candidates=50,
    max_block_size=500,
):
    t_entity = target_df["entity_id"].array
    t_name = target_df["norm_business_name"].array
    t_address = target_df[
        "norm_business_address"
    ].array
    t_number = target_df["street_number"].array
    t_country = target_df["norm_country"].array
    t_city = target_df["norm_city"].array
    t_state = target_df["norm_state"].array
    t_postal = target_df[
        "norm_postal_code"
    ].array

    print(
        "PASS 1/3 - counting base blocks",
        flush=True,
    )

    block_counts = defaultdict(int)
    countries = defaultdict(set)

    for i in range(len(target_df)):
        country = as_text(t_country[i])

        keys = base_keys(
            t_name[i],
            t_address[i],
            t_number[i],
        )

        for kind, key in keys:
            block = (
                kind,
                key,
                country,
            )

            block_counts[block] += 1

            countries[
                (kind, key)
            ].add(country)

        if (
            (i + 1) % 500000
            == 0
        ):
            print(
                f"Counted "
                f"{i + 1:,}/"
                f"{len(target_df):,}",
                flush=True,
            )

    oversized_blocks = {
        block
        for block, count
        in block_counts.items()
        if count > max_block_size
    }

    print(
        f"Oversized base blocks: "
        f"{len(oversized_blocks):,}",
        flush=True,
    )

    print(
        "PASS 2/3 - building indexes",
        flush=True,
    )

    normal_index = defaultdict(list)

    # Full membership only for oversized blocks.
    # uint32 is dramatically cheaper than Python ints.
    oversized_members = {}

    # Refined buckets are capped at max_block_size + 1.
    refined_index = defaultdict(list)

    for i in range(len(target_df)):
        country = as_text(t_country[i])

        base = base_keys(
            t_name[i],
            t_address[i],
            t_number[i],
        )

        refinements = None

        for kind, key in base:
            block = (
                kind,
                key,
                country,
            )

            if block not in oversized_blocks:
                normal_index[
                    block
                ].append(i)

                continue

            members = oversized_members.get(
                block
            )

            if members is None:
                members = array("I")
                oversized_members[
                    block
                ] = members

            members.append(i)

            if refinements is None:
                refinements = refinement_keys(
                    t_name[i],
                    t_address[i],
                    t_number[i],
                    t_city[i],
                    t_state[i],
                    t_postal[i],
                )

            for ref_kind, ref_key in refinements:
                refined_block = (
                    kind,
                    key,
                    country,
                    ref_kind,
                    ref_key,
                )

                bucket = refined_index[
                    refined_block
                ]

                # We only need to know:
                # <= max_block_size or oversized.
                if len(bucket) <= max_block_size:
                    bucket.append(i)

        if (
            (i + 1) % 500000
            == 0
        ):
            print(
                f"Indexed "
                f"{i + 1:,}/"
                f"{len(target_df):,}",
                flush=True,
            )

    print(
        "PASS 3/3 - processing sources",
        flush=True,
    )

    s_entity = source_df["entity_id"].array
    s_name = source_df["norm_business_name"].array
    s_address = source_df[
        "norm_business_address"
    ].array
    s_number = source_df["street_number"].array
    s_country = source_df["norm_country"].array
    s_city = source_df["norm_city"].array
    s_state = source_df["norm_state"].array
    s_postal = source_df[
        "norm_postal_code"
    ].array

    total_pairs = 0

    sources_without_candidates = 0
    sources_capped = 0

    normal_bucket_hits = 0
    refined_bucket_hits = 0
    oversized_fallback_hits = 0
    oversized_queries = 0

    true_pairs = 0
    retained_true_pairs = 0

    if truth is not None:
        true_pairs = sum(
            len(ids)
            for ids in truth.values()
        )

    with open(
        output_path,
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        handle.write(
            "source1_entity_id\t"
            "candidate_entity_ids\n"
        )

        for source_idx in range(
            len(source_df)
        ):
            source_id = as_text(
                s_entity[source_idx]
            )

            source_name = as_text(
                s_name[source_idx]
            )

            source_address = as_text(
                s_address[source_idx]
            )

            source_number = as_text(
                s_number[source_idx]
            )

            country = as_text(
                s_country[source_idx]
            )

            source_base = base_keys(
                source_name,
                source_address,
                source_number,
            )

            source_refinements = (
                refinement_keys(
                    source_name,
                    source_address,
                    source_number,
                    s_city[source_idx],
                    s_state[source_idx],
                    s_postal[source_idx],
                )
            )

            candidate_indices = set()

            # Full oversized blocks that could not be
            # reduced sufficiently. We rank them later
            # instead of discarding them.
            fallback_blocks = set()

            for kind, key in sorted(
                source_base
            ):
                if country:
                    scopes = (
                        country,
                        "",
                    )
                else:
                    scopes = tuple(
                        countries.get(
                            (kind, key),
                            (),
                        )
                    )

                for scope in sorted(scopes):
                    block = (
                        kind,
                        key,
                        scope,
                    )

                    count = block_counts.get(
                        block,
                        0,
                    )

                    if count == 0:
                        continue

                    if count <= max_block_size:
                        bucket = normal_index.get(
                            block,
                            (),
                        )

                        candidate_indices.update(
                            bucket
                        )

                        normal_bucket_hits += 1

                        continue

                    oversized_queries += 1

                    found_refinement = False

                    for (
                        ref_kind,
                        ref_key,
                    ) in source_refinements:
                        refined_block = (
                            kind,
                            key,
                            scope,
                            ref_kind,
                            ref_key,
                        )

                        bucket = refined_index.get(
                            refined_block
                        )

                        if not bucket:
                            continue

                        # max_block_size + 1 means oversized.
                        if (
                            len(bucket)
                            > max_block_size
                        ):
                            continue

                        candidate_indices.update(
                            bucket
                        )

                        refined_bucket_hits += 1
                        found_refinement = True

                    # Zero-drop fallback:
                    # never simply discard the original
                    # oversized block.
                    if not found_refinement:
                        fallback_blocks.add(
                            block
                        )

            def ranking_tuple(target_idx):
                target_name = as_text(
                    t_name[target_idx]
                )

                target_address = as_text(
                    t_address[target_idx]
                )

                if (
                    source_name
                    and target_name
                ):
                    name_score = (
                        fuzz.token_sort_ratio(
                            source_name,
                            target_name,
                        )
                    )
                else:
                    name_score = 0.0

                if (
                    source_address
                    and target_address
                ):
                    address_score = (
                        fuzz.token_sort_ratio(
                            source_address,
                            target_address,
                        )
                    )
                else:
                    address_score = 0.0

                score = (
                    0.7 * name_score
                    + 0.3 * address_score
                )

                return (
                    score,
                    as_text(
                        t_entity[target_idx]
                    ),
                )

            # Rank the regular + refined candidates.
            initial_count = len(
                candidate_indices
            )

            if (
                initial_count
                > max_candidates
            ):
                sources_capped += 1

            # Keep a bounded top list before expensive
            # oversized fallback scanning.
            best = heapq.nlargest(
                max_candidates,
                candidate_indices,
                key=ranking_tuple,
            )

            best_set = set(best)

            # If an oversized bucket had NO usable
            # refinement, rank the complete bucket
            # rather than dropping it.
            #
            # Memory remains bounded because only the
            # best max_candidates items survive.
            for block in fallback_blocks:
                members = oversized_members.get(
                    block,
                    ()
                )

                if not members:
                    continue

                oversized_fallback_hits += 1

                pool = list(best_set)

                for target_idx in members:
                    if target_idx in best_set:
                        continue

                    pool.append(target_idx)

                    # Periodically shrink so pool does
                    # not grow with the bucket.
                    if (
                        len(pool)
                        >= max_candidates * 20
                    ):
                        pool = heapq.nlargest(
                            max_candidates,
                            pool,
                            key=ranking_tuple,
                        )

                        best_set = set(pool)

                best = heapq.nlargest(
                    max_candidates,
                    pool,
                    key=ranking_tuple,
                )

                best_set = set(best)

            # Final deterministic order.
            best.sort(
                key=lambda idx: (
                    -ranking_tuple(idx)[0],
                    ranking_tuple(idx)[1],
                )
            )

            target_ids = [
                as_text(t_entity[idx])
                for idx in best[
                    :max_candidates
                ]
            ]

            total_pairs += len(
                target_ids
            )

            if not target_ids:
                sources_without_candidates += 1

            if truth is not None:
                expected = truth.get(
                    source_id,
                    set(),
                )

                retained_true_pairs += len(
                    expected
                    & set(target_ids)
                )

            handle.write(
                source_id
                + "\t"
                + ",".join(
                    sorted(target_ids)
                )
                + "\n"
            )

            if (
                (source_idx + 1)
                % 100000
                == 0
            ):
                print(
                    f"Sources "
                    f"{source_idx + 1:,}/"
                    f"{len(source_df):,} | "
                    f"pairs={total_pairs:,}",
                    flush=True,
                )

    result = {
        "pairs": total_pairs,
        "sources_without_candidates":
            sources_without_candidates,
        "sources_capped":
            sources_capped,
        "oversized_queries":
            oversized_queries,
        "normal_bucket_hits":
            normal_bucket_hits,
        "refined_bucket_hits":
            refined_bucket_hits,
        "oversized_fallback_hits":
            oversized_fallback_hits,
    }

    if truth is not None:
        result.update(
            {
                "true_pairs":
                    true_pairs,

                "retained_true_pairs":
                    retained_true_pairs,

                "blocking_recall":
                    (
                        retained_true_pairs
                        / true_pairs
                        if true_pairs
                        else None
                    ),
            }
        )

    return result
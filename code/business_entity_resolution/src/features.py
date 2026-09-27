"""ML-2: ordered, versioned feature contract shared with persisted models."""
import numpy as np
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


def compute_pairwise_features(
    pairs_df,
    s1_df,
    tgt_df,
    chunk_size=100000,
):
    if pairs_df.empty:
        return pd.DataFrame(
            columns=PAIR_COLUMNS + FEATURE_COLUMNS
        )

    if s1_df.entity_id.duplicated().any():
        raise ValueError(
            "Feature lookup requires unique source entity IDs"
        )

    if tgt_df.entity_id.duplicated().any():
        raise ValueError(
            "Feature lookup requires unique target entity IDs"
        )

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

    # ---------------------------------------------------------
    # DO NOT convert millions of rows to Python dictionaries.
    #
    # Pandas Index gives us ID -> integer-position lookup while
    # keeping the actual records inside the existing DataFrames.
    # ---------------------------------------------------------

    source_index = pd.Index(
        s1_df["entity_id"],
        copy=False,
    )

    target_index = pd.Index(
        tgt_df["entity_id"],
        copy=False,
    )

    # References to existing columns.
    s_name = s1_df["norm_business_name"].to_numpy(copy=False)
    s_address = s1_df["norm_business_address"].to_numpy(copy=False)
    s_number = s1_df["street_number"].to_numpy(copy=False)
    s_city = s1_df["norm_city"].to_numpy(copy=False)
    s_state = s1_df["norm_state"].to_numpy(copy=False)
    s_postal = s1_df["norm_postal_code"].to_numpy(copy=False)
    s_country = s1_df["norm_country"].to_numpy(copy=False)

    t_name = tgt_df["norm_business_name"].to_numpy(copy=False)
    t_address = tgt_df["norm_business_address"].to_numpy(copy=False)
    t_number = tgt_df["street_number"].to_numpy(copy=False)
    t_city = tgt_df["norm_city"].to_numpy(copy=False)
    t_state = tgt_df["norm_state"].to_numpy(copy=False)
    t_postal = tgt_df["norm_postal_code"].to_numpy(copy=False)
    t_country = tgt_df["norm_country"].to_numpy(copy=False)

    total_pairs = len(pairs_df)

    # float32 cuts feature-memory usage in half compared with
    # the normal float64 representation.
    feature_values = np.empty(
        (total_pairs, len(FEATURE_COLUMNS)),
        dtype=np.float32,
    )

    source_ids = pairs_df[
        PAIR_COLUMNS[0]
    ]

    target_ids = pairs_df[
        PAIR_COLUMNS[1]
    ]

    # ---------------------------------------------------------
    # Process pair lookups in bounded chunks.
    # ---------------------------------------------------------

    for start in range(0, total_pairs, chunk_size):

        end = min(
            start + chunk_size,
            total_pairs,
        )

        source_positions = source_index.get_indexer(
            source_ids.iloc[start:end]
        )

        target_positions = target_index.get_indexer(
            target_ids.iloc[start:end]
        )

        missing_source = source_positions == -1
        missing_target = target_positions == -1

        if missing_source.any() or missing_target.any():
            bad_offset = np.flatnonzero(
                missing_source | missing_target
            )[0]

            row = start + int(bad_offset)

            raise ValueError(
                "Candidate references unknown entity: "
                f"{source_ids.iloc[row]}, "
                f"{target_ids.iloc[row]}"
            )

        for offset in range(end - start):

            row = start + offset

            source_pos = source_positions[offset]
            target_pos = target_positions[offset]

            n1 = as_text(
                s_name[source_pos]
            )

            n2 = as_text(
                t_name[target_pos]
            )

            a1 = as_text(
                s_address[source_pos]
            )

            a2 = as_text(
                t_address[target_pos]
            )

            num1 = as_text(
                s_number[source_pos]
            )

            num2 = as_text(
                t_number[target_pos]
            )

            city1 = as_text(
                s_city[source_pos]
            )

            city2 = as_text(
                t_city[target_pos]
            )

            state1 = as_text(
                s_state[source_pos]
            )

            state2 = as_text(
                t_state[target_pos]
            )

            postal1 = as_text(
                s_postal[source_pos]
            )

            postal2 = as_text(
                t_postal[target_pos]
            )

            country1 = as_text(
                s_country[source_pos]
            )

            country2 = as_text(
                t_country[target_pos]
            )

            names = bool(n1 and n2)
            addresses = bool(a1 and a2)

            feature_values[row, 0] = (
                fuzz.token_sort_ratio(n1, n2) / 100
                if names
                else 0.0
            )

            feature_values[row, 1] = (
                fuzz.token_set_ratio(n1, n2) / 100
                if names
                else 0.0
            )

            feature_values[row, 2] = (
                JaroWinkler.normalized_similarity(
                    n1,
                    n2,
                )
                if names
                else 0.0
            )

            feature_values[row, 3] = (
                fuzz.ratio(n1, n2) / 100
                if names
                else 0.0
            )

            feature_values[row, 4] = (
                jaccard(n1, n2)
                if names
                else 0.0
            )

            feature_values[row, 5] = (
                fuzz.ratio(a1, a2) / 100
                if addresses
                else 0.0
            )

            feature_values[row, 6] = (
                fuzz.token_sort_ratio(
                    a1,
                    a2,
                ) / 100
                if addresses
                else 0.0
            )

            feature_values[row, 7] = (
                jaccard(a1, a2)
                if addresses
                else 0.0
            )

            feature_values[row, 8] = exact(
                num1,
                num2,
            )

            feature_values[row, 9] = float(
                bool(num1 and num2)
                and num1 != num2
            )

            feature_values[row, 10] = exact(
                city1,
                city2,
            )

            feature_values[row, 11] = exact(
                state1,
                state2,
            )

            feature_values[row, 12] = exact(
                postal1,
                postal2,
            )

            feature_values[row, 13] = exact(
                country1,
                country2,
            )

            feature_values[row, 14] = float(
                names
            )

            feature_values[row, 15] = float(
                addresses
            )

        print(
            f"Features: {end:,}/{total_pairs:,} "
            f"({100 * end / total_pairs:.1f}%)",
            flush=True,
        )

    # ---------------------------------------------------------
    # Add numeric columns directly onto the existing pairs
    # DataFrame rather than creating another giant list of rows.
    # ---------------------------------------------------------

    for column_index, column_name in enumerate(
        FEATURE_COLUMNS
    ):
        pairs_df[column_name] = feature_values[
            :,
            column_index,
        ]

    return pairs_df
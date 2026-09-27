"""ML-1: conservative, Unicode-aware name and address normalization."""
import re
import unicodedata
import pandas as pd
from data_io import TEXT_COLUMNS

LEGAL_SUFFIXES = {"inc", "incorporated", "llc", "ltd", "limited", "corp", "corporation", "pvt", "private", "plc", "llp"}
ADDRESS_WORDS = {"st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "blvd": "boulevard", "ln": "lane", "hwy": "highway", "apt": "apartment", "ste": "suite", "fl": "floor"}
COUNTRIES = {"us": "united states", "usa": "united states", "u s a": "united states", "uk": "united kingdom", "u k": "united kingdom", "in": "india"}


def clean_text(text):
    if pd.isna(text):
        return ""
    text = unicodedata.normalize("NFKD", str(text).casefold())
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = text.replace("&", " and ")
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s]|_", " ", text)).strip()


def clean_name(text):
    tokens = clean_text(text).split()
    while tokens and tokens[-1] in LEGAL_SUFFIXES:
        tokens.pop()
    if tokens and tokens[0] == "the":
        tokens.pop(0)
    return " ".join(tokens)


def clean_address(text):
    return " ".join(ADDRESS_WORDS.get(t, t) for t in clean_text(text).split())


def street_number(address):
    # Later numbers may be a ZIP code or unit; use only the leading number.
    match = re.match(r"^(\d+[a-z]?)\b", address)
    return match.group(1) if match else ""


def normalize_dataset(df):
    result = df.copy()
    for col in TEXT_COLUMNS:
        values = result[col] if col in result else pd.Series("", index=result.index)
        fn = clean_name if col == "business_name" else clean_address if col == "business_address" else clean_text
        result[f"norm_{col}"] = values.map(fn)
    result["norm_country"] = result.norm_country.map(lambda c: COUNTRIES.get(c, c))
    result["street_number"] = result.norm_business_address.map(street_number)
    return result

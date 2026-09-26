import re
import pandas as pd

def clean_text(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in ["business_name", "business_address", "country"]:
        if col in df.columns:
            df[f"norm_{col}"] = df[col].apply(clean_text)
        else:
            df[f"norm_{col}"] = ""
    return df
import json
import re
from pathlib import Path

import pandas as pd

def load_merchant_mapping():
    mapping_path = (
        Path(__file__).parent.parent
        / "config"
        / "merchant_mapping.json"
    )

    with open(mapping_path, "r") as file:
        return json.load(file)

MERCHANT_MAPPING = load_merchant_mapping()

def preprocess_transactions(df):
    df = df.copy()

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df["Amount"] = pd.to_numeric(df["Amount"], errors="coerce")

    df = df.dropna(subset=["Date", "Description", "Amount"])

    df["Description"] = df["Description"].astype(str).str.strip()
    df = df[df["Description"] != ""]

    df["Merchant"] = df["Description"].apply(normalize_merchant)

    if "Category" in df.columns:
        df["Category"] = df["Category"].astype(str).str.strip()
    
    columns = [
    "Date",
    "Description",
    "Merchant",
    "Amount",
    ]

    if "Category" in df.columns:
        columns.append("Category")

    return df[columns].reset_index(drop=True)




def normalize_merchant(description):
    description = description.lower()

    for merchant, keywords in MERCHANT_MAPPING.items():
        for keyword in keywords:
            if keyword.lower() in description:
                return merchant

    cleaned = re.sub(r"[^a-z0-9 ]", " ", description)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    return cleaned.upper()

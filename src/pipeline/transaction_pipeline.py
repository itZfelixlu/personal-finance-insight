import hashlib

import pandas as pd

from src.ml.categorizer import categorize_transactions
from src.pipeline.data_loader import load_transactions_csv
from src.pipeline.preprocessing import preprocess_transactions


def _process_uploaded_file(uploaded_file):
    raw_df = load_transactions_csv(uploaded_file)
    clean_df = preprocess_transactions(raw_df)

    # Synthetic data contains ground-truth labels. Drop them so both synthetic
    # and real statements go through the same production model.
    clean_df = clean_df.drop(columns=["Category"], errors="ignore")

    categorized_df = categorize_transactions(clean_df)
    categorized_df["Source File"] = uploaded_file.name
    return categorized_df


def _file_digest(uploaded_file):
    """Return a content hash without changing the file's read position."""
    if hasattr(uploaded_file, "getvalue"):
        content = uploaded_file.getvalue()
    else:
        position = uploaded_file.tell()
        content = uploaded_file.read()
        uploaded_file.seek(position)

    if isinstance(content, str):
        content = content.encode("utf-8")

    return hashlib.sha256(content).hexdigest()


def combine_uploaded_files(uploaded_files):
    dataframes = []
    errors = []
    warnings = []
    seen_files = {}

    for uploaded_file in uploaded_files:
        digest = _file_digest(uploaded_file)
        original_name = seen_files.get(digest)

        if original_name is not None:
            warnings.append(
                f"Skipped duplicate file {uploaded_file.name}; "
                f"it is identical to {original_name}."
            )
            continue

        seen_files[digest] = uploaded_file.name

        try:
            dataframes.append(_process_uploaded_file(uploaded_file))
        except Exception as error:
            errors.append(f"Could not process {uploaded_file.name}: {error}")

    if not dataframes:
        return pd.DataFrame(), errors, warnings

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df["Month"] = combined_df["Date"].dt.to_period("M").astype(str)
    return combined_df, errors, warnings

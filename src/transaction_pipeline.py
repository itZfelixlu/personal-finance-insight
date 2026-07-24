import pandas as pd

from src.categorizer import categorize_transactions
from src.data_loader import load_transactions_csv
from src.preprocessing import preprocess_transactions


def _process_uploaded_file(uploaded_file):
    """Clean and categorize one uploaded Chase statement."""
    raw_df = load_transactions_csv(uploaded_file)
    clean_df = preprocess_transactions(raw_df)

    # Synthetic data contains ground-truth labels. Drop them so both synthetic
    # and real statements go through the same production model.
    clean_df = clean_df.drop(columns=["Category"], errors="ignore")

    categorized_df = categorize_transactions(clean_df)
    categorized_df["Source File"] = uploaded_file.name
    return categorized_df


def combine_uploaded_files(uploaded_files):
    """Process valid files and collect user-facing errors separately."""
    dataframes = []
    errors = []

    for uploaded_file in uploaded_files:
        try:
            dataframes.append(_process_uploaded_file(uploaded_file))
        except Exception as error:
            errors.append(f"Could not process {uploaded_file.name}: {error}")

    if not dataframes:
        return pd.DataFrame(), errors

    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df["Month"] = combined_df["Date"].dt.to_period("M").astype(str)
    return combined_df, errors

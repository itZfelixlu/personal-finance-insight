import pandas as pd

DATA_COLUMN_CANDIDATES = ("Date", "Posting Date", "Transaction Date")
REQUIRED_COLUMNS = ("Date", "Description", "Amount")

def load_transactions_csv(file_path):
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    if "Date" not in df.columns:
        for column in DATA_COLUMN_CANDIDATES:
            if column in df.columns:
                df = df.rename(columns={column: "Date"})
                break
    
    missing_columns = [
        column for column in REQUIRED_COLUMNS if column not in df.columns
    ]

    if missing_columns:
        missing = ", ".join(missing_columns)
        available = ", ".join(df.columns)
        raise ValueError(
            f"Missing required transaction columns: {missing}. "
            f"Available columns: {available}"
        )

    return df

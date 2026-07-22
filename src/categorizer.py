"""Transaction categorizer (production inference).

Loads the trained model exported by the experiment notebook
(`models/category_model.pkl`) and adds a `Category` column to a clean
transaction DataFrame. This module NEVER trains a model — it only loads and
predicts, per the project architecture.

Input : clean DataFrame with columns Date, Description, Merchant, Amount
Output: same DataFrame + a Category column
"""

from functools import lru_cache
from pathlib import Path

import joblib

MODEL_PATH = Path(__file__).parent.parent / "models" / "category_model.pkl"

# The model was trained on the transaction Description text, so we predict from
# the same column here (train/inference must use the same input).
TEXT_COLUMN = "Description"


@lru_cache(maxsize=1)
def load_category_model(model_path=None):
    """Load and cache the trained category model.

    Cached so repeated calls (e.g. per Streamlit rerun) don't re-read the file.
    """
    path = Path(model_path) if model_path else MODEL_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Category model not found at {path}. "
            "Train and export it first by running notebooks/experiments.ipynb."
        )
    return joblib.load(path)


def categorize_transactions(df, model=None):
    """Return a copy of `df` with a predicted `Category` column added.

    Parameters
    ----------
    df : pd.DataFrame
        Clean transactions containing a `Description` column.
    model : optional
        A pre-loaded model (mainly for testing). Defaults to the exported model.
    """
    if TEXT_COLUMN not in df.columns:
        raise ValueError(
            f"Expected a '{TEXT_COLUMN}' column for categorization. "
            f"Available columns: {', '.join(df.columns)}"
        )

    if model is None:
        model = load_category_model()

    df = df.copy()
    df["Category"] = model.predict(df[TEXT_COLUMN].astype(str))
    return df

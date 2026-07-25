"""Weekly feature engineering (Member B — Unsupervised Learning).

Aggregates a *categorized* transaction DataFrame into one feature vector per
calendar week. These vectors are the input to the KMeans spending-pattern model
(`pattern_predictor.py`).

This module is deliberately the **single source of truth** for how a week is
turned into numbers: the experiment notebook (training) and the production app
(inference) both import from here, so the model always sees features built the
exact same way. The feature ordering is also frozen in
`models/feature_config.json`.

Input  : DataFrame with columns Date, Description, Amount, Category
Output : DataFrame with one row per week and the feature columns below.

Feature set (matches ARCHITECTURE.md + the project proposal)
------------------------------------------------------------
Composition ("占比" / ratios of weekly spend, each in [0, 1]):
    dining_ratio      share of the week's spend on Dining
    grocery_ratio     share on Grocery
    shopping_ratio    share on Shopping
    transport_ratio   share on Transport
Magnitude:
    total_spend       total money spent that week      ("总额")
    txn_count         number of spending transactions  ("笔数")
    avg_txn           mean amount per spending txn      ("平均")
"""

from pathlib import Path

import pandas as pd

# --- Configuration ----------------------------------------------------------
# Categories that get their own composition ratio. Kept explicit (not derived
# from the data) so a week that happens to have no e.g. Transport still produces
# a transport_ratio column of 0.0 -- the feature vector must have a fixed shape.
RATIO_CATEGORIES = ["Dining", "Grocery", "Shopping", "Transport"]

# The frozen feature order. Training and inference MUST use this exact order.
FEATURE_COLUMNS = (
    [f"{c.lower()}_ratio" for c in RATIO_CATEGORIES]
    + ["total_spend", "txn_count", "avg_txn"]
)

# How weeks are anchored. "W-SUN" = weeks ending Sunday (a common statement
# convention). Change in one place only.
WEEK_RULE = "W-SUN"

REQUIRED_COLUMNS = ("Date", "Amount", "Category")


def _spend_series(amount: pd.Series) -> pd.Series:
    """Money *going out* only, as a positive number.

    Debits are negative in the Chase export; credits/refunds/incoming transfers
    are positive. Spending pattern discovery cares about outflow, so incoming
    money is mapped to 0 rather than counted as negative spend.
    """
    return (-amount).clip(lower=0)


def build_weekly_features(df: pd.DataFrame) -> pd.DataFrame:
    """Turn categorized transactions into one feature row per week.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain Date (datetime-like), Amount (numeric) and Category.

    Returns
    -------
    pd.DataFrame
        Columns: ``Week`` (the week-start date) followed by every column in
        :data:`FEATURE_COLUMNS`, sorted by week. Weeks with no outgoing spend
        are dropped (they carry no spending signal).
    """
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"weekly features need columns {list(REQUIRED_COLUMNS)}; "
            f"missing: {missing}. Present: {list(df.columns)}"
        )

    df = df.copy()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])

    df["Spend"] = _spend_series(pd.to_numeric(df["Amount"], errors="coerce"))
    df = df.dropna(subset=["Spend"])

    # Week bucket -> its start date (a clean, sortable, human-readable key).
    df["Week"] = df["Date"].dt.to_period(WEEK_RULE).apply(lambda p: p.start_time)

    rows = []
    for week, g in df.groupby("Week"):
        total = float(g["Spend"].sum())
        if total <= 0:
            continue  # no spending signal this week

        spend_txns = g[g["Spend"] > 0]
        count = int(len(spend_txns))

        # Per-category spend for the ratio features.
        cat_spend = g.groupby("Category")["Spend"].sum()

        row = {"Week": week}
        for cat in RATIO_CATEGORIES:
            row[f"{cat.lower()}_ratio"] = float(cat_spend.get(cat, 0.0)) / total
        row["total_spend"] = total
        row["txn_count"] = count
        row["avg_txn"] = total / count if count else 0.0
        rows.append(row)

    out = pd.DataFrame(rows, columns=["Week"] + FEATURE_COLUMNS)
    return out.sort_values("Week").reset_index(drop=True)


def load_feature_config(path=None) -> dict:
    """Load the frozen feature ordering exported during training."""
    if path is None:
        path = Path(__file__).parents[2] / "models" / "feature_config.json"
    import json

    with open(path) as f:
        return json.load(f)


def feature_matrix(weekly_df: pd.DataFrame, feature_order=None):
    """Select feature columns in the canonical order -> 2-D array for the model.

    Using an explicit order (from ``feature_config.json`` at inference time)
    guarantees the scaler/KMeans receive columns in the same positions they saw
    during training.
    """
    order = feature_order or FEATURE_COLUMNS
    missing = [c for c in order if c not in weekly_df.columns]
    if missing:
        raise ValueError(f"weekly feature frame missing columns: {missing}")
    return weekly_df[order].to_numpy(dtype=float)

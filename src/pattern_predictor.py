"""Weekly spending-pattern predictor (Member B - Unsupervised Learning).

Production inference for the unsupervised half of the pipeline. Loads the
artifacts exported by the experiment notebook:

    models/weekly_scaler.pkl          StandardScaler fitted on training weeks
    models/spending_pattern_model.pkl KMeans model
    models/cluster_names.json         cluster id -> human-readable name
    models/feature_config.json        frozen feature ordering

and assigns each weekly feature vector to a spending pattern. This module NEVER
fits a scaler or trains KMeans -- it only loads and predicts, per the project
architecture.

Input : Weekly Feature DataFrame (from weekly_features.build_weekly_features)
Output: same rows + PatternID and PatternName columns
"""

import json
from functools import lru_cache
from pathlib import Path

import joblib

from src.weekly_features import feature_matrix

MODELS_DIR = Path(__file__).parent.parent / "models"
SCALER_PATH = MODELS_DIR / "weekly_scaler.pkl"
MODEL_PATH = MODELS_DIR / "spending_pattern_model.pkl"
NAMES_PATH = MODELS_DIR / "cluster_names.json"
CONFIG_PATH = MODELS_DIR / "feature_config.json"


@lru_cache(maxsize=1)
def load_pattern_artifacts():
    """Load and cache the scaler, KMeans model, cluster names and feature order.

    Returns
    -------
    (scaler, model, cluster_names, feature_order)
        ``cluster_names`` maps int cluster id -> str name.
        ``feature_order`` is the list of feature columns used at training time.
    """
    for path in (SCALER_PATH, MODEL_PATH, NAMES_PATH, CONFIG_PATH):
        if not path.exists():
            raise FileNotFoundError(
                f"Missing pattern artifact: {path}. Train and export it first "
                "by running the clustering section of notebooks/experiments.ipynb."
            )

    scaler = joblib.load(SCALER_PATH)
    model = joblib.load(MODEL_PATH)

    with open(NAMES_PATH) as f:
        # JSON keys are strings; normalize to int cluster ids.
        cluster_names = {int(k): v for k, v in json.load(f).items()}

    with open(CONFIG_PATH) as f:
        feature_order = json.load(f)["feature_columns"]

    return scaler, model, cluster_names, feature_order


def predict_patterns(weekly_df, artifacts=None):
    """Add ``PatternID`` and ``PatternName`` columns to a weekly feature frame.

    Parameters
    ----------
    weekly_df : pd.DataFrame
        Output of ``weekly_features.build_weekly_features`` (has the feature
        columns, and usually a ``Week`` column).
    artifacts : optional
        Pre-loaded ``(scaler, model, cluster_names, feature_order)`` tuple,
        mainly for testing. Defaults to the exported artifacts.
    """
    if artifacts is None:
        artifacts = load_pattern_artifacts()
    scaler, model, cluster_names, feature_order = artifacts

    if weekly_df.empty:
        out = weekly_df.copy()
        out["PatternID"] = []
        out["PatternName"] = []
        return out

    X = feature_matrix(weekly_df, feature_order)
    X_scaled = scaler.transform(X)
    labels = model.predict(X_scaled)

    out = weekly_df.copy()
    out["PatternID"] = labels.astype(int)
    out["PatternName"] = [cluster_names.get(int(c), f"Cluster {int(c)}") for c in labels]
    return out

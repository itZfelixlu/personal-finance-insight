"""Train & export the weekly spending-pattern model (Member B).

Mirrors the clustering section of notebooks/experiments.ipynb so the model
artifacts can be regenerated headlessly. Produces:

    models/weekly_scaler.pkl
    models/spending_pattern_model.pkl
    models/cluster_names.json
    models/feature_config.json

Usage:
    python scripts/train_patterns.py
    python scripts/train_patterns.py --k 4
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from src.pipeline.data_loader import load_transactions_csv
from src.pipeline.preprocessing import preprocess_transactions
from src.ml.weekly_features import (
    FEATURE_COLUMNS,
    RATIO_CATEGORIES,
    build_weekly_features,
    feature_matrix,
)

DATA_FILE = PROJECT_ROOT / "inputs" / "synthetic_chase_statement_2025.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RANDOM_STATE = 42

# Which scaled-feature signature maps to which human-readable pattern name.
FEATURE_LABELS = {
    "dining_ratio": "Dining-heavy",
    "grocery_ratio": "Grocery-focused",
    "shopping_ratio": "Shopping-heavy",
    "transport_ratio": "Transport-heavy",
    "total_spend": "High-Spending",
    "txn_count": "High-Frequency",
    "avg_txn": "Big-Ticket",
}
BALANCED_THRESHOLD = 0.5  # if no feature's z-score exceeds this, call it Balanced
TIE_MARGIN = 0.4          # z-scores within this are treated as a "tie"
RATIO_FEATURES = {f"{c.lower()}_ratio" for c in RATIO_CATEGORIES}


def name_clusters(kmeans, feature_columns, mean_ratios):
    """Assign a readable name to each cluster from its standardized centroid.

    KMeans centroids live in standardized space, so each coordinate is already a
    z-score: how far the cluster sits from the average week on that feature. A
    cluster is named after its most *distinctive* (highest positive z) feature.

    Tie-break: some ratio features (e.g. transport) have tiny variance, so a
    small absolute lift produces a large z-score and could win the name despite
    representing only a sliver of actual spend. When the top two features are
    within ``TIE_MARGIN`` z and both are composition ratios, we prefer the one
    that is the larger *share of spend* (``mean_ratios``) — so a week that is 14%
    dining / 6% transport is called "Dining-heavy", which is what a user expects.

    Falls back to "Balanced" when nothing stands out; collisions get a qualifier.
    """
    centers = kmeans.cluster_centers_
    names = {}
    used = set()
    for cid, z in enumerate(centers):
        order = list(np.argsort(z)[::-1])  # feature indices by descending z
        top, second = order[0], order[1]
        top_feat, second_feat = feature_columns[top], feature_columns[second]

        if z[top] < BALANCED_THRESHOLD:
            name = "Balanced"
        else:
            chosen = top_feat
            # near-tie between two composition ratios -> pick the bigger share
            if (
                z[top] - z[second] < TIE_MARGIN
                and top_feat in RATIO_FEATURES
                and second_feat in RATIO_FEATURES
                and mean_ratios[cid][second_feat] > mean_ratios[cid][top_feat]
            ):
                chosen = second_feat
            name = FEATURE_LABELS[chosen]

        base, i = name, 2
        while name in used:
            name = f"{base} {i}"
            i += 1
        used.add(name)
        names[cid] = name
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=4, help="number of clusters")
    ap.add_argument("--data", default=str(DATA_FILE))
    args = ap.parse_args()

    MODELS_DIR.mkdir(exist_ok=True)

    # 1. Load -> preprocess -> weekly features (same modules as the app)
    df = preprocess_transactions(load_transactions_csv(args.data))
    weekly = build_weekly_features(df)
    X = feature_matrix(weekly)
    print(f"Weeks: {len(weekly)} | Features ({len(FEATURE_COLUMNS)}): {FEATURE_COLUMNS}")

    # 2. Standardize (features are on wildly different scales: ratios ~[0,1] vs
    #    total_spend in the thousands). KMeans uses Euclidean distance, so
    #    without scaling total_spend would dominate every cluster.
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    # 3. Model selection sweep (reported; K fixed by --k)
    print("\nModel selection (elbow + silhouette):")
    print(f"{'K':<3}{'inertia':<12}{'silhouette':<12}")
    for k in range(2, 9):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit(Xs)
        print(f"{k:<3}{km.inertia_:<12.2f}{silhouette_score(Xs, km.labels_):<12.3f}")

    # 4. Final model
    model = KMeans(n_clusters=args.k, random_state=RANDOM_STATE, n_init=10).fit(Xs)
    sil = silhouette_score(Xs, model.labels_)
    print(f"\nFinal K={args.k} | silhouette={sil:.3f}")

    # 5. Name clusters and report profiles
    # mean_ratios[cid][feat] = that cluster's average value for each ratio feature
    tmp = weekly.copy()
    tmp["_c"] = model.labels_
    grp = tmp.groupby("_c")[FEATURE_COLUMNS].mean()
    mean_ratios = {cid: grp.loc[cid].to_dict() for cid in grp.index}
    names = name_clusters(model, FEATURE_COLUMNS, mean_ratios)
    weekly = weekly.copy()
    weekly["PatternID"] = model.labels_
    weekly["PatternName"] = weekly["PatternID"].map(names)
    profile = weekly.groupby("PatternName")[FEATURE_COLUMNS].mean()
    profile["weeks"] = weekly.groupby("PatternName").size()
    print("\nCluster profiles (original units):")
    print(profile.round(3).to_string())

    # 6. Export artifacts
    joblib.dump(scaler, MODELS_DIR / "weekly_scaler.pkl")
    joblib.dump(model, MODELS_DIR / "spending_pattern_model.pkl")
    with open(MODELS_DIR / "cluster_names.json", "w") as f:
        json.dump({str(k): v for k, v in names.items()}, f, indent=2)
    with open(MODELS_DIR / "feature_config.json", "w") as f:
        json.dump(
            {
                "feature_columns": FEATURE_COLUMNS,
                "ratio_categories": RATIO_CATEGORIES,
                "n_clusters": args.k,
                "silhouette": round(float(sil), 3),
                "scaler": "StandardScaler",
                "model": "KMeans",
                "n_training_weeks": int(len(weekly)),
            },
            f,
            indent=2,
        )
    print("\nExported: weekly_scaler.pkl, spending_pattern_model.pkl, "
          "cluster_names.json, feature_config.json")


if __name__ == "__main__":
    main()

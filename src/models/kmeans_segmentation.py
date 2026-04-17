"""
src/models/kmeans_segmentation.py
─────────────────────────────────
K-means clustering for game segmentation based on features or sales patterns.
"""
import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def fit_kmeans(
    X: np.ndarray,
    n_clusters: int = 3,
    random_state: int = 42,
    verbose: bool = False,
) -> Tuple[KMeans, np.ndarray]:
    """
    Fit K-means clustering on the feature matrix.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (samples × features)
    n_clusters : int
        Number of clusters (default 3)
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress messages

    Returns
    -------
    kmeans : KMeans
        Fitted K-means model
    labels : np.ndarray
        Cluster labels for each sample
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10,
        verbose=1 if verbose else 0,
    )
    labels = kmeans.fit_predict(X_scaled)

    if verbose:
        logger.info(f"K-means fitted with {n_clusters} clusters")
        for i in range(n_clusters):
            count = (labels == i).sum()
            logger.info(f"  Cluster {i}: {count} samples ({100*count/len(labels):.1f}%)")

    return kmeans, labels


def segment_by_publisher_class(
    df: pd.DataFrame,
    publisher_col: str = "publisher_class_ord",
) -> Tuple[dict, dict]:
    """
    Create segments based on publisher class (simpler alternative to K-means).

    Parameters
    ----------
    df : pd.DataFrame
        Data with publisher class column
    publisher_col : str
        Name of the publisher class column

    Returns
    -------
    segment_dict : dict
        Mapping of segment name to boolean mask
    segment_labels : dict
        Mapping of segment name to class value
    """
    segments = {}
    labels = {}

    class_mapping = {0: "Hobbyist", 1: "Indie", 2: "AA", 3: "AAA"}

    for val, name in class_mapping.items():
        if publisher_col in df.columns:
            mask = df[publisher_col] == val
            segments[name] = mask
            labels[name] = val
        else:
            logger.warning(f"Column {publisher_col} not found in DataFrame")

    return segments, labels


def segment_by_price(
    df: pd.DataFrame,
    price_col: str = "Price",
    bins: list = None,
) -> Tuple[dict, list]:
    """
    Create price-based market segments.

    Parameters
    ----------
    df : pd.DataFrame
        Data with price column
    price_col : str
        Name of the price column
    bins : list
        Price cutoffs for segments. Default: [0, 5, 20, 60, 1000]

    Returns
    -------
    segments : dict
        Mapping of segment name to boolean mask
    bin_edges : list
        The bin edges used
    """
    if bins is None:
        bins = [0, 5, 20, 60, 1000]

    segment_names = ["Free", "Budget (<$5)", "Mid-tier ($5-$20)", "Premium ($20-$60)", "AAA ($60+)"]
    segments = {}

    for i in range(len(bins) - 1):
        mask = (df[price_col] >= bins[i]) & (df[price_col] < bins[i + 1])
        segment_names_i = segment_names[i] if i < len(segment_names) else f"${bins[i]}-${bins[i+1]}"
        segments[segment_names_i] = mask

    return segments, bins


def get_segment_statistics(
    df: pd.DataFrame,
    segments: dict,
    target_col: str = "copiesSold",
) -> pd.DataFrame:
    """
    Compute summary statistics for each segment.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset
    segments : dict
        Mapping of segment name to boolean mask
    target_col : str
        Target column name

    Returns
    -------
    stats_df : pd.DataFrame
        Statistics by segment
    """
    records = []

    for seg_name, mask in segments.items():
        seg_data = df[mask][target_col].dropna()

        records.append({
            "segment": seg_name,
            "n_games": mask.sum(),
            "pct_games": 100 * mask.sum() / len(df),
            "mean_sales": seg_data.mean(),
            "median_sales": seg_data.median(),
            "std_sales": seg_data.std(),
            "min_sales": seg_data.min(),
            "max_sales": seg_data.max(),
        })

    return pd.DataFrame(records)


def elbow_analysis(
    X: np.ndarray,
    k_range: list = None,
    random_state: int = 42,
) -> Tuple[list, list]:
    """
    Run K-means for different values of k and return inertias (elbow method).

    Parameters
    ----------
    X : np.ndarray
        Feature matrix
    k_range : list
        Range of k values to test. Default: [2, 3, 4, 5, 6, 7, 8, 9, 10]
    random_state : int
        Random seed

    Returns
    -------
    k_values : list
        Tested k values
    inertias : list
        Inertia for each k
    """
    if k_range is None:
        k_range = list(range(2, 11))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    inertias = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=random_state, n_init=10)
        kmeans.fit(X_scaled)
        inertias.append(kmeans.inertia_)
        logger.info(f"K={k}: Inertia={kmeans.inertia_:.2f}")

    return k_range, inertias


def analyze_kmeans_clusters(
    X: np.ndarray,
    labels: np.ndarray,
    feature_names: list,
    df_raw: pd.DataFrame = None,
    target_col: str = "copiesSold",
) -> pd.DataFrame:
    """
    Analyze and describe each K-means cluster by feature means and target statistics.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (already scaled/transformed)
    labels : np.ndarray
        Cluster labels from K-means
    feature_names : list
        Names of features in X
    df_raw : pd.DataFrame, optional
        Raw data for computing target statistics
    target_col : str
        Target column name

    Returns
    -------
    cluster_profile : pd.DataFrame
        Cluster profiles with size, target stats, and top 3 features
    """
    n_clusters = len(np.unique(labels))
    records = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id
        cluster_size = mask.sum()
        pct = 100 * cluster_size / len(labels)

        # Feature importance for this cluster
        X_cluster = X[mask]
        feature_means = np.abs(X_cluster.mean(axis=0))  # Absolute mean for importance
        top_feature_idx = np.argsort(feature_means)[-3:][::-1]
        top_features = ", ".join([feature_names[i] for i in top_feature_idx])

        record = {
            "Cluster": cluster_id,
            "Size": cluster_size,
            "Pct": f"{pct:.1f}%",
            "Top_Features": top_features,
        }

        # Target statistics if raw data provided
        if df_raw is not None and target_col in df_raw.columns:
            targets = df_raw[target_col].iloc[mask].dropna()
            record.update({
                "Mean_Sales": targets.mean(),
                "Median_Sales": targets.median(),
                "Std_Sales": targets.std(),
            })

        records.append(record)

    return pd.DataFrame(records)

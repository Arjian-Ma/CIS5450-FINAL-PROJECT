"""
src/models/baseline_model_visualization.py
───────────────────────────────────────────
Visualization functions for baseline and advanced model predictions.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_actual_vs_predicted_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 20,
    random_state: int = 42,
    figsize: tuple = (14, 6),
    title: str = "Actual vs Predicted (20 Random Test Games)",
) -> plt.Figure:
    """
    Plot actual vs predicted log1p(copiesSold) for n_samples random games
    from the test set as a side-by-side bar chart.

    Parameters
    ----------
    y_true : np.ndarray
        Actual log1p(copiesSold) values from test set
    y_pred : np.ndarray
        Predicted log1p(copiesSold) values from test set
    n_samples : int
        Number of random games to display (default 20)
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size (width, height)
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    np.random.seed(random_state)

    # Sample n_samples random indices
    n_total = len(y_true)
    indices = np.random.choice(n_total, size=min(n_samples, n_total), replace=False)
    indices = np.sort(indices)

    y_true_sample = y_true[indices]
    y_pred_sample = y_pred[indices]

    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        "Game": [f"Game {i+1}" for i in range(len(indices))],
        "Actual": y_true_sample,
        "Predicted": y_pred_sample,
    })

    # Compute residuals and errors
    df["Residual"] = df["Predicted"] - df["Actual"]
    df["Error"] = np.abs(df["Residual"])

    # Plot
    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(len(df))
    width = 0.35

    bars1 = ax.bar(x - width/2, df["Actual"], width, label="Actual", alpha=0.8, color="steelblue")
    bars2 = ax.bar(x + width/2, df["Predicted"], width, label="Predicted", alpha=0.8, color="coral")

    ax.set_xlabel("Game Sample", fontsize=11)
    ax.set_ylabel("log1p(copiesSold)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(df["Game"], rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    return fig, df


def plot_residual_scatter_sample(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 20,
    random_state: int = 42,
    figsize: tuple = (10, 6),
    title: str = "Residuals vs Predicted (20 Random Test Games)",
) -> plt.Figure:
    """
    Scatter plot of residuals (predicted - actual) vs predicted values
    for n_samples random games.

    Parameters
    ----------
    y_true : np.ndarray
        Actual log1p(copiesSold) values
    y_pred : np.ndarray
        Predicted log1p(copiesSold) values
    n_samples : int
        Number of random games to display
    random_state : int
        Random seed for reproducibility
    figsize : tuple
        Figure size
    title : str
        Plot title

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object
    """
    np.random.seed(random_state)

    indices = np.random.choice(len(y_true), size=min(n_samples, len(y_true)), replace=False)

    y_true_sample = y_true[indices]
    y_pred_sample = y_pred[indices]
    residuals = y_pred_sample - y_true_sample

    fig, ax = plt.subplots(figsize=figsize)

    scatter = ax.scatter(
        y_pred_sample, residuals,
        s=100, alpha=0.6, c=np.abs(residuals),
        cmap="RdYlGn_r", edgecolors="black", linewidth=0.5
    )

    ax.axhline(0, color="red", linestyle="--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Predicted log1p(copiesSold)", fontsize=11)
    ax.set_ylabel("Residual (Predicted - Actual)", fontsize=11)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, label="Absolute Error")
    ax.legend(fontsize=10)

    plt.tight_layout()

    return fig


def model_prediction_summary(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_samples: int = 20,
) -> pd.DataFrame:
    """
    Return a summary DataFrame of actual vs predicted for n_samples.

    Parameters
    ----------
    y_true : np.ndarray
        Actual values
    y_pred : np.ndarray
        Predicted values
    n_samples : int
        Number of samples

    Returns
    -------
    summary_df : pd.DataFrame
        DataFrame with Actual, Predicted, Residual, and Error columns
    """
    np.random.seed(42)
    indices = np.random.choice(len(y_true), size=min(n_samples, len(y_true)), replace=False)
    indices = np.sort(indices)

    df = pd.DataFrame({
        "Game_ID": indices,
        "Actual": y_true[indices],
        "Predicted": y_pred[indices],
    })
    df["Residual"] = df["Predicted"] - df["Actual"]
    df["Abs_Error"] = np.abs(df["Residual"])

    return df.round(4)

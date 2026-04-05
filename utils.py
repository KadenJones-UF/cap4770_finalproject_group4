import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    ConfusionMatrixDisplay,
)

RAW_DATA_PATH = "data/amazon_products.csv"
DROP_COLS = ["asin", "imgUrl", "productURL", "title"]
FEATURE_COLS = ["stars", "reviews", "price", "listPrice", "category_id", "boughtInLastMonth"]
TARGET_COL = "isBestSeller"
RANDOM_STATE = 42


def load_data(path: str = RAW_DATA_PATH) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    df = df[df["stars"] > 0]
    df = df.dropna()
    df[TARGET_COL] = df[TARGET_COL].astype(int)
    return df


def get_X_y(df: pd.DataFrame):
    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy()
    return X, y


def split_data(X, y, test_size: float = 0.2):
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def get_metrics(y_true, y_pred, model_name) -> dict:
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall    = recall_score(y_true, y_pred, zero_division=0)
    f1        = f1_score(y_true, y_pred, zero_division=0)
    roc_auc   = roc_auc_score(y_true, y_pred)

    return {
        "model": model_name,
        "precision": precision,
        "recall": recall,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "roc_auc": roc_auc,
        "confusion_matrix": cm,
    }


def plot_confusion_matrix(cm, model_name: str, save_path: str = None):
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Bestseller", "Bestseller"],
    )
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix — {model_name}")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_metric_comparison(results: list[dict], save_path: str = None):
    metrics = ["precision", "recall", "specificity", "f1", "roc_auc"]
    labels  = [r["model"] for r in results]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, result in enumerate(results):
        vals = [result[m] for m in metrics]
        ax.bar(x + i * width, vals, width, label=result["model"])

    ax.set_xlabel("Metric")
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison — Key Metrics")
    ax.set_xticks(x + width * (len(results) - 1) / 2)
    ax.set_xticklabels(["Precision", "Recall\n(Sensitivity)", "Specificity", "F1", "ROC-AUC"])
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()

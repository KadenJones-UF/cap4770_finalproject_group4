import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from utils import (
    get_metrics,
    load_data,
    get_X_y,
    split_data,
    plot_confusion_matrix,
    plot_metric_comparison,
    FEATURE_COLS,
    RANDOM_STATE,
)


def build_logistic_regression() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
            solver="lbfgs",
        )),
    ])


def build_random_forest() -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )),
    ])


def plot_feature_importance(rf_pipeline: Pipeline, save_path: str = None):
    importances = rf_pipeline.named_steps["clf"].feature_importances_
    indices     = np.argsort(importances)[::-1]
    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_vals     = importances[indices]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_vals[::-1], color="forestgreen")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest — Feature Importances")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()



def plot_lr_coefficients(lr_pipeline: Pipeline, save_path: str = None):
    coefs   = lr_pipeline.named_steps["clf"].coef_[0]
    indices = np.argsort(np.abs(coefs))[::-1]

    sorted_features = [FEATURE_COLS[i] for i in indices]
    sorted_coefs    = coefs[indices]
    colors          = ["steelblue" if c > 0 else "salmon" for c in sorted_coefs]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(sorted_features[::-1], sorted_coefs[::-1], color=colors[::-1])
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Coefficient (scaled)")
    ax.set_title("Logistic Regression — Feature Coefficients")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def main():
    df = load_data()

    X, y = get_X_y(df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    results = []

    lr = build_logistic_regression()
    lr.fit(X_train, y_train)

    y_pred_lr = lr.predict(X_test)
    lr_metrics = get_metrics(y_test, y_pred_lr, "Logistic Regression")
    results.append(lr_metrics)

    plot_confusion_matrix(
        lr_metrics["confusion_matrix"],
        model_name="Logistic Regression",
        save_path="cm_logistic_regression.png",
    )
    plot_lr_coefficients(lr, save_path="lr_coefficients.png")

    rf = build_random_forest()
    rf.fit(X_train, y_train)

    y_pred_rf = rf.predict(X_test)
    rf_metrics = get_metrics(y_test, y_pred_rf, "Random Forest")
    results.append(rf_metrics)

    plot_confusion_matrix(
        rf_metrics["confusion_matrix"],
        model_name="Random Forest",
        save_path="cm_random_forest.png",
    )
    plot_feature_importance(rf, save_path="rf_feature_importance.png")
    plot_metric_comparison(results, save_path="model_comparison.png")

if __name__ == "__main__":
    main()

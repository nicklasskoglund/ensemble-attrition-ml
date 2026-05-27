"""
src/evaluation.py
=================
Handles model evaluation, visualisation and result persistence for the
Employee Attrition Prediction project.

Functions
---------
evaluate_model()             – computes metrics and saves results to JSON.
plot_confusion_matrix()      – heatmap of predicted vs actual labels.
plot_roc_curve()             – ROC curve with AUC score.
plot_feature_importance()    – built-in tree feature importances (bar chart).
plot_permutation_importance()– model-agnostic importance via feature shuffling.
compare_models()             – side-by-side bar chart of all model metrics.

Why these metrics?
------------------
The dataset is imbalanced (~16 % Attrition = Yes). Accuracy is misleading –
a model that always predicts "No" scores 84 % without being useful.

Primary metric : F1-score (Yes)  – balances Precision and Recall.
Secondary      : ROC-AUC         – ranks risk levels across thresholds.
Business focus : Recall (Yes)    – minimise missed at-risk employees.

Usage Example
-------------
    from src.evaluation import evaluate_model, plot_roc_curve

    results = evaluate_model(model, X_test, y_test, model_name="random_forest")
    plot_roc_curve(model, X_test, y_test, model_name="random_forest", save=True)
"""

import json
import os

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


# ── Constants ──────────────────────────────────────────────────────────────────

RESULTS_DIR = "outputs/results"
FIGURES_DIR = "outputs/figures"

# Reproducibility seed for permutation importance.
RANDOM_STATE = 42

# Default number of top features shown in importance plots.
TOP_N_FEATURES = 20


# ── Internal helpers ───────────────────────────────────────────────────────────

def _ensure_dirs() -> None:
    """Create output directories if they do not exist."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)


def _save_figure(fig: plt.Figure, filename: str) -> None:
    """
    Save a matplotlib figure to the figures output directory.

    Parameters
    ----------
    fig      : plt.Figure
    filename : str
        File name without path, e.g. ``"roc_random_forest.png"``.
    """
    _ensure_dirs()
    path = os.path.join(FIGURES_DIR, filename)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"[evaluation] Figure saved to: {path}")


# ── 1. Evaluate ────────────────────────────────────────────────────────────────

def evaluate_model(
    model,
    X_test,
    y_test,
    model_name: str,
) -> dict:
    """
    Compute evaluation metrics and persist results to JSON.

    Metrics computed
    ----------------
    - F1-score (positive class = Attrition Yes)
    - ROC-AUC
    - Full classification report (precision, recall, f1 per class)

    The JSON file is written to ``outputs/results/<model_name>.json`` so
    that the story notebook (06) can load results dynamically without
    re-running training.

    Parameters
    ----------
    model      : fitted sklearn-compatible classifier
    X_test     : pd.DataFrame
    y_test     : pd.Series
    model_name : str
        Used as filename stem and display label, e.g. ``"random_forest"``.

    Returns
    -------
    dict
        Dictionary with keys: model, f1, roc_auc, report.
    """
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    f1      = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    report  = classification_report(y_test, y_pred, output_dict=True)

    results = {
        "model":   model_name,
        "f1":      round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "report":  report,
    }

    _ensure_dirs()
    out_path = os.path.join(RESULTS_DIR, f"{model_name}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"[evaluation] {model_name} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    print(f"[evaluation] Results saved to: {out_path}")
    return results


# ── 2. Confusion Matrix ────────────────────────────────────────────────────────

def plot_confusion_matrix(
    model,
    X_test,
    y_test,
    model_name: str,
    save: bool = False,
) -> None:
    """
    Plot a heatmap of the confusion matrix.

    Rows represent actual labels, columns represent predicted labels.
    Useful for visually inspecting false negatives (missed attrition cases)
    vs false positives (unnecessary HR interventions).

    Parameters
    ----------
    model      : fitted classifier
    X_test     : pd.DataFrame
    y_test     : pd.Series
    model_name : str
    save       : bool, optional
        If True, save the figure to outputs/figures/. Default: False.
    """
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["No Attrition", "Attrition"],
    )
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(f"Confusion Matrix – {model_name.replace('_', ' ').title()}")
    plt.tight_layout()

    if save:
        _save_figure(fig, f"cm_{model_name}.png")
    plt.show()


# ── 3. ROC Curve ──────────────────────────────────────────────────────────────

def plot_roc_curve(
    model,
    X_test,
    y_test,
    model_name: str,
    save: bool = False,
) -> None:
    """
    Plot the ROC curve with AUC score annotated.

    The ROC curve shows the trade-off between True Positive Rate (Recall)
    and False Positive Rate across all classification thresholds. A model
    with no skill follows the diagonal (AUC = 0.5).

    Parameters
    ----------
    model      : fitted classifier
    X_test     : pd.DataFrame
    y_test     : pd.Series
    model_name : str
    save       : bool, optional
        If True, save the figure to outputs/figures/. Default: False.
    """
    y_proba = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    auc = roc_auc_score(y_test, y_proba)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="No skill")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate (Recall)")
    ax.set_title(f"ROC Curve – {model_name.replace('_', ' ').title()}")
    ax.legend(loc="lower right")
    plt.tight_layout()

    if save:
        _save_figure(fig, f"roc_{model_name}.png")
    plt.show()


# ── 4. Feature Importance ─────────────────────────────────────────────────────

def plot_feature_importance(
    model,
    feature_names,
    model_name: str,
    top_n: int = TOP_N_FEATURES,
    save: bool = False,
) -> None:
    """
    Plot the built-in feature importances of a tree-based model.

    Uses ``model.feature_importances_`` (mean decrease in impurity).
    Note: this measure can be biased towards high-cardinality features.
    Use ``plot_permutation_importance()`` for a model-agnostic alternative.

    Parameters
    ----------
    model        : fitted tree-based classifier (RF or XGBoost)
    feature_names: list[str] or pd.Index
        Column names of the training features.
    model_name   : str
    top_n        : int, optional
        Number of top features to display. Default: 20.
    save         : bool, optional
        If True, save the figure to outputs/figures/. Default: False.
    """
    importances = model.feature_importances_
    indices     = np.argsort(importances)[-top_n:]
    names       = np.array(feature_names)[indices]
    values      = importances[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(names, values, color="steelblue")
    ax.set_xlabel("Importance (mean decrease in impurity)")
    ax.set_title(
        f"Feature Importance (Top {top_n}) – {model_name.replace('_', ' ').title()}"
    )
    plt.tight_layout()

    if save:
        _save_figure(fig, f"fi_{model_name}.png")
    plt.show()


# ── 5. Permutation Importance ─────────────────────────────────────────────────

def plot_permutation_importance(
    model,
    X_test,
    y_test,
    model_name: str,
    top_n: int = TOP_N_FEATURES,
    save: bool = False,
) -> None:
    """
    Plot permutation importance using boxplots.

    For each feature, its values are randomly shuffled ``n_repeats`` times.
    The drop in F1-score measures how much the model relied on that feature.
    A feature with near-zero permutation importance can be removed without
    harming performance – even if its built-in importance appears high.

    This is the preferred importance method for final interpretation because
    it is model-agnostic and evaluated on held-out test data.

    Parameters
    ----------
    model      : fitted classifier
    X_test     : pd.DataFrame
    y_test     : pd.Series
    model_name : str
    top_n      : int, optional
        Number of top features to display. Default: 20.
    save       : bool, optional
        If True, save the figure to outputs/figures/. Default: False.
    """
    print(f"[evaluation] Computing permutation importance for {model_name} (this may take a moment)...")
    result  = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        scoring="f1",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    indices = np.argsort(result.importances_mean)[-top_n:]
    names   = np.array(X_test.columns)[indices]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.boxplot(
        result.importances[indices].T,
        vert=False,
        labels=names,
    )
    ax.axvline(0, color="grey", linestyle="--", lw=1)
    ax.set_xlabel("Decrease in F1-score")
    ax.set_title(
        f"Permutation Importance (Top {top_n}) – {model_name.replace('_', ' ').title()}"
    )
    plt.tight_layout()

    if save:
        _save_figure(fig, f"pi_{model_name}.png")
    plt.show()


# ── 6. Model Comparison ───────────────────────────────────────────────────────

def compare_models(results: list[dict], save: bool = False) -> None:
    """
    Plot a grouped bar chart comparing F1 and ROC-AUC across models.

    Loads pre-computed result dictionaries (returned by ``evaluate_model()``)
    and renders them side-by-side for easy comparison in the story notebook.

    Parameters
    ----------
    results : list[dict]
        List of result dicts from ``evaluate_model()``.
        Each dict must contain keys: ``model``, ``f1``, ``roc_auc``.
    save    : bool, optional
        If True, save the figure to outputs/figures/. Default: False.

    Example
    -------
        compare_models([dt_results, rf_results, xgb_results], save=True)
    """
    model_names = [r["model"].replace("_", " ").title() for r in results]
    f1_scores   = [r["f1"]      for r in results]
    auc_scores  = [r["roc_auc"] for r in results]

    x     = np.arange(len(model_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5))
    bars_f1  = ax.bar(x - width / 2, f1_scores,  width, label="F1-score",  color="steelblue")
    bars_auc = ax.bar(x + width / 2, auc_scores, width, label="ROC-AUC",   color="coral")

    # Annotate bars with values
    for bar in bars_f1 + bars_auc:
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{bar.get_height():.3f}",
            ha="center", va="bottom", fontsize=9,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names)
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison – F1-score vs ROC-AUC")
    ax.legend()
    plt.tight_layout()

    if save:
        _save_figure(fig, "model_comparison.png")
    plt.show()
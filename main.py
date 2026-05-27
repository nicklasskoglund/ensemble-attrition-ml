"""
main.py
=======
End-to-end pipeline for the Employee Attrition Prediction project.

Running this script will:
    1. Load and clean the raw IBM HR dataset.
    2. Save the cleaned dataset to data/attrition_clean.csv.
    3. Split data into train / test sets.
    4. Train and evaluate a Decision Tree baseline.
    5. Train, tune and evaluate a Random Forest.
    6. Train, tune and evaluate an XGBoost classifier.
    7. Plot and save Feature Importance for RF and XGBoost.
    8. Plot and save Permutation Importance for RF and XGBoost.
    9. Save a side-by-side model comparison chart.

All results are saved to:
    outputs/results/  ← JSON metric files
    outputs/figures/  ← PNG plots

Usage
-----
    # Activate virtual environment first
    .venv\\Scripts\\activate          # Windows
    source .venv/bin/activate        # macOS / Linux

    python main.py
"""

import sys

from src.data_processing import clean_and_encode, load_raw, save_clean, split_and_scale
from src.evaluation import (
    compare_models,
    evaluate_model,
    plot_confusion_matrix,
    plot_feature_importance,
    plot_permutation_importance,
    plot_roc_curve,
)
from src.model_training import (
    train_decision_tree,
    train_random_forest,
    train_xgboost,
    tune_random_forest,
    tune_xgboost,
)

# ── Config ─────────────────────────────────────────────────────────────────────

RAW_DATA_PATH   = "data/WA_Fn-UseC_-HR-Employee-Attrition.csv"
CLEAN_DATA_PATH = "data/attrition_clean.csv"


# ── Pipeline steps ─────────────────────────────────────────────────────────────

def step_data() -> tuple:
    """
    Step 1 – Load, clean and split data.

    Returns
    -------
    X_train, X_test, y_train, y_test
    """
    print("\n" + "=" * 60)
    print("STEP 1 – DATA")
    print("=" * 60)

    df_raw   = load_raw(RAW_DATA_PATH)
    df_clean = clean_and_encode(df_raw)
    save_clean(df_clean, CLEAN_DATA_PATH)

    X_train, X_test, y_train, y_test = split_and_scale(df_clean)
    return X_train, X_test, y_train, y_test


def step_decision_tree(X_train, X_test, y_train, y_test) -> dict:
    """
    Step 2 – Baseline Decision Tree.

    Returns
    -------
    dict
        Evaluation results.
    """
    print("\n" + "=" * 60)
    print("STEP 2 – DECISION TREE (BASELINE)")
    print("=" * 60)

    model   = train_decision_tree(X_train, y_train)
    results = evaluate_model(model, X_test, y_test, model_name="decision_tree")

    plot_confusion_matrix(model, X_test, y_test, model_name="decision_tree", save=True)
    plot_roc_curve(model, X_test, y_test, model_name="decision_tree", save=True)

    return results


def step_random_forest(X_train, X_test, y_train, y_test) -> dict:
    """
    Step 3 – Random Forest: baseline train → tune → evaluate.

    Returns
    -------
    dict
        Evaluation results for the tuned model.
    """
    print("\n" + "=" * 60)
    print("STEP 3 – RANDOM FOREST")
    print("=" * 60)

    # 3a. Train with default params first (quick sanity check)
    print("\n-- 3a. Default Random Forest --")
    rf_default = train_random_forest(X_train, y_train)
    evaluate_model(rf_default, X_test, y_test, model_name="random_forest_default")

    # 3b. Hyperparameter tuning
    print("\n-- 3b. Tuning Random Forest (GridSearchCV) --")
    rf_tuned, best_params = tune_random_forest(X_train, y_train)
    results = evaluate_model(rf_tuned, X_test, y_test, model_name="random_forest")

    # 3c. Plots
    plot_confusion_matrix(rf_tuned, X_test, y_test, model_name="random_forest", save=True)
    plot_roc_curve(rf_tuned, X_test, y_test, model_name="random_forest", save=True)
    plot_feature_importance(
        rf_tuned, X_train.columns, model_name="random_forest", save=True
    )
    plot_permutation_importance(
        rf_tuned, X_test, y_test, model_name="random_forest", save=True
    )

    return results


def step_xgboost(X_train, X_test, y_train, y_test) -> dict:
    """
    Step 4 – XGBoost: baseline train → tune → evaluate.

    Returns
    -------
    dict
        Evaluation results for the tuned model.
    """
    print("\n" + "=" * 60)
    print("STEP 4 – XGBOOST")
    print("=" * 60)

    # 4a. Train with default params first
    print("\n-- 4a. Default XGBoost --")
    xgb_default = train_xgboost(X_train, y_train)
    evaluate_model(xgb_default, X_test, y_test, model_name="xgboost_default")

    # 4b. Hyperparameter tuning
    print("\n-- 4b. Tuning XGBoost (RandomizedSearchCV) --")
    xgb_tuned, best_params = tune_xgboost(X_train, y_train)
    results = evaluate_model(xgb_tuned, X_test, y_test, model_name="xgboost")

    # 4c. Plots
    plot_confusion_matrix(xgb_tuned, X_test, y_test, model_name="xgboost", save=True)
    plot_roc_curve(xgb_tuned, X_test, y_test, model_name="xgboost", save=True)
    plot_feature_importance(
        xgb_tuned, X_train.columns, model_name="xgboost", save=True
    )
    plot_permutation_importance(
        xgb_tuned, X_test, y_test, model_name="xgboost", save=True
    )

    return results


def step_comparison(dt_results: dict, rf_results: dict, xgb_results: dict) -> None:
    """
    Step 5 – Side-by-side model comparison chart.

    Parameters
    ----------
    dt_results  : dict  Decision Tree evaluation results.
    rf_results  : dict  Random Forest evaluation results.
    xgb_results : dict  XGBoost evaluation results.
    """
    print("\n" + "=" * 60)
    print("STEP 5 – MODEL COMPARISON")
    print("=" * 60)

    compare_models(
        [dt_results, rf_results, xgb_results],
        save=True,
    )


# ── Entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    """Run the full pipeline end-to-end."""

    print("\n" + "=" * 60)
    print("  Employee Attrition Prediction – Full Pipeline")
    print("=" * 60)

    # Verify raw data exists before starting
    import os
    if not os.path.exists(RAW_DATA_PATH):
        print(
            f"\n[ERROR] Raw data not found at: {RAW_DATA_PATH}\n"
            "Please download WA_Fn-UseC_-HR-Employee-Attrition.csv from Kaggle\n"
            "and place it in the data/ folder. See README.md for instructions.\n"
        )
        sys.exit(1)

    X_train, X_test, y_train, y_test = step_data()

    dt_results  = step_decision_tree(X_train, X_test, y_train, y_test)
    rf_results  = step_random_forest(X_train, X_test, y_train, y_test)
    xgb_results = step_xgboost(X_train, X_test, y_train, y_test)

    step_comparison(dt_results, rf_results, xgb_results)

    print("\n" + "=" * 60)
    print("  Pipeline complete.")
    print(f"  Results  → outputs/results/")
    print(f"  Figures  → outputs/figures/")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
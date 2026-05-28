"""
src/model_training.py
=====================
Handles training and hyperparameter tuning for all models in the
Employee Attrition Prediction project.

Models
------
- Decision Tree  : Baseline – simple, interpretable, prone to overfitting.
- Random Forest  : Ensemble bagging – reduces variance vs a single tree.
- XGBoost        : Ensemble boosting – sequentially corrects errors.

Tuning
------
- Random Forest  : GridSearchCV   – exhaustive search over a defined grid.
- XGBoost        : RandomizedSearchCV – efficient sampling over a wider space.

Both tuners use F1-score as the optimisation metric (preferred over accuracy
given the ~16/84 class imbalance in the dataset).

Usage Example
-------------
    from src.data_processing import load_raw, clean_and_encode, split_and_scale
    from src.model_training import train_random_forest, tune_xgboost

    df    = clean_and_encode(load_raw("data/WA_Fn-UseC_-HR-Employee-Attrition.csv"))
    X_train, X_test, y_train, y_test = split_and_scale(df)

    rf            = train_random_forest(X_train, y_train)
    xgb, params   = tune_xgboost(X_train, y_train)
"""

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from xgboost import XGBClassifier


# ── Constants ──────────────────────────────────────────────────────────────────

# Reproducibility seed used across the project.
RANDOM_STATE = 42

# Cross-validation folds used in all tuning functions.
CV_FOLDS = 5

# Primary optimisation metric – F1 on the positive class (Attrition = Yes).
SCORING = "f1"


# ── Helper ─────────────────────────────────────────────────────────────────────

def _class_weight_scale(y_train):
    """
    Compute scale_pos_weight for XGBoost.

    XGBoost uses this ratio to up-weight the minority class (Attrition = Yes)
    during training, compensating for the ~16/84 class imbalance.

    Returns
    -------
    float
        Number of negative samples / number of positive samples.
    """
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    return n_neg / n_pos


# ── 1. Decision Tree ───────────────────────────────────────────────────────────

def train_decision_tree(X_train, y_train) -> DecisionTreeClassifier:
    """
    Train a single Decision Tree classifier.

    Used as a **baseline** to illustrate the limitations of a single tree
    (high variance, tendency to overfit) before introducing ensemble methods.

    ``class_weight='balanced'`` adjusts for the class imbalance by weighting
    each class inversely proportional to its frequency.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels (0 = No, 1 = Yes).

    Returns
    -------
    DecisionTreeClassifier
        Fitted model.
    """
    print("[model_training] Training Decision Tree (baseline)...")
    model = DecisionTreeClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )
    model.fit(X_train, y_train)
    print("[model_training] Decision Tree training complete.")
    return model


# ── 2. Random Forest ───────────────────────────────────────────────────────────

def train_random_forest(
    X_train,
    y_train,
    params: dict = None,
) -> RandomForestClassifier:
    """
    Train a Random Forest classifier.

    Builds an ensemble of decision trees using **bagging** (Bootstrap
    AGGregatING). Each tree is trained on a random subset of rows and
    features, reducing overfitting compared to a single tree.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    params  : dict, optional
        Override any default hyperparameters (e.g. from a tuning run).

    Returns
    -------
    RandomForestClassifier
        Fitted model.
    """
    defaults = dict(
        n_estimators=200,        # number of trees in the forest
        class_weight="balanced", # handle class imbalance
        n_jobs=-1,               # use all available CPU cores
        random_state=RANDOM_STATE,
    )
    if params:
        defaults.update(params)

    print(f"[model_training] Training Random Forest with params: {defaults}")
    model = RandomForestClassifier(**defaults)
    model.fit(X_train, y_train)
    print("[model_training] Random Forest training complete.")
    return model


def tune_random_forest(X_train, y_train):
    """
    Tune Random Forest hyperparameters using GridSearchCV.

    Performs an exhaustive search over the parameter grid below, evaluated
    with {CV_FOLDS}-fold cross-validation and F1-score as the target metric.

    Search space
    ------------
    n_estimators      : number of trees
    max_depth         : maximum depth of each tree (None = unlimited)
    min_samples_split : minimum samples required to split an internal node
    max_features      : number of features considered at each split

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    best_model  : RandomForestClassifier
        Refitted on full training set with best parameters.
    best_params : dict
        The winning hyperparameter combination.
    """
    param_grid = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [None, 10, 20],
        "min_samples_split": [2, 5, 10],
        "max_features":      ["sqrt", "log2"],
    }

    base = RandomForestClassifier(
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print(f"[model_training] GridSearchCV – Random Forest | folds={CV_FOLDS} | scoring={SCORING}")
    gs = GridSearchCV(
        estimator=base,
        param_grid=param_grid,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        verbose=1,
    )
    gs.fit(X_train, y_train)

    print(f"[model_training] Best RF params : {gs.best_params_}")
    print(f"[model_training] Best RF F1 (CV): {gs.best_score_:.4f}")
    return gs.best_estimator_, gs.best_params_


# ── 3. XGBoost ─────────────────────────────────────────────────────────────────

def train_xgboost(
    X_train,
    y_train,
    params: dict = None,
) -> XGBClassifier:
    """
    Train an XGBoost classifier.

    Uses **gradient boosting** – trees are built sequentially, each one
    correcting the residual errors of the previous, producing a strong
    learner from many weak ones.

    ``scale_pos_weight`` is computed automatically from the training labels
    to compensate for the class imbalance.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features.
    y_train : pd.Series
        Training labels.
    params  : dict, optional
        Override any default hyperparameters.

    Returns
    -------
    XGBClassifier
        Fitted model.
    """
    defaults = dict(
        n_estimators=200,
        scale_pos_weight=_class_weight_scale(y_train),  # handle class imbalance
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    if params:
        defaults.update(params)

    print(f"[model_training] Training XGBoost with params: {defaults}")
    model = XGBClassifier(**defaults)
    model.fit(X_train, y_train)
    print("[model_training] XGBoost training complete.")
    return model


def tune_xgboost(X_train, y_train):
    """
    Tune XGBoost hyperparameters using RandomizedSearchCV.

    Samples ``n_iter=30`` random combinations from the parameter
    distributions below, evaluated with {CV_FOLDS}-fold cross-validation
    and F1-score as the target metric.

    RandomizedSearchCV is preferred over GridSearchCV here because the
    XGBoost search space is larger – random sampling finds good regions
    faster than exhaustive enumeration.

    Search space
    ------------
    n_estimators     : number of boosting rounds
    max_depth        : maximum depth of each tree
    learning_rate    : step size shrinkage (lower = more robust, slower)
    subsample        : fraction of training rows sampled per tree
    colsample_bytree : fraction of features sampled per tree

    Parameters
    ----------
    X_train : pd.DataFrame
    y_train : pd.Series

    Returns
    -------
    best_model  : XGBClassifier
        Refitted on full training set with best parameters.
    best_params : dict
        The winning hyperparameter combination.
    """
    param_dist = {
        "n_estimators":      [100, 200, 300],
        "max_depth":         [3, 5, 7],
        "learning_rate":     [0.01, 0.05, 0.1, 0.2],
        "subsample":         [0.7, 0.8, 1.0],
        "colsample_bytree":  [0.7, 0.8, 1.0],
    }

    base = XGBClassifier(
        scale_pos_weight=_class_weight_scale(y_train),
        eval_metric="logloss",
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )

    print(f"[model_training] RandomizedSearchCV – XGBoost | folds={CV_FOLDS} | scoring={SCORING} | n_iter=30")
    rs = RandomizedSearchCV(
        estimator=base,
        param_distributions=param_dist,
        n_iter=30,
        cv=CV_FOLDS,
        scoring=SCORING,
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    rs.fit(X_train, y_train)

    print(f"[model_training] Best XGB params : {rs.best_params_}")
    print(f"[model_training] Best XGB F1 (CV): {rs.best_score_:.4f}")
    return rs.best_estimator_, rs.best_params_
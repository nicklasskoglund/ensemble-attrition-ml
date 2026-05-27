"""
src/data_processing.py
======================
Handles all data loading, cleaning, encoding and splitting for the
Employee Attrition Prediction project.

Pipeline
--------
1. load_raw()         – reads the raw Kaggle CSV
2. clean_and_encode() – drops constants, encodes target + categoricals
3. save_clean()       – saves cleaned DataFrame to data/attrition_clean.csv
4. split_and_scale()  – train/test split with optional StandardScaler

Usage Example
-------------
    from src.data_processing import load_raw, clean_and_encode, save_clean, split_and_scale

    df_raw   = load_raw("data/WA_Fn-UseC_-HR-Employee-Attrition.csv")
    df_clean = clean_and_encode(df_raw)
    save_clean(df_clean)
    X_train, X_test, y_train, y_test = split_and_scale(df_clean)
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ── Constants ──────────────────────────────────────────────────────────────────

# These columns carry no information: every row has the same value.
CONSTANT_COLUMNS = ["EmployeeCount", "Over18", "StandardHours"]

# Dropped because it is just a row identifier, not a feature.
ID_COLUMN = "EmployeeNumber"

# The column we want to predict.
TARGET = "Attrition"

# Raw CSV encoding (the Kaggle file is UTF-8).
CSV_ENCODING = "utf-8"

# Reproducibility seed used across the project.
RANDOM_STATE = 42


# ── 1. Load ────────────────────────────────────────────────────────────────────

def load_raw(path: str) -> pd.DataFrame:
    """
    Read the raw IBM HR CSV file from disk.

    Parameters
    ----------
    path : str
        Path to WA_Fn-UseC_-HR-Employee-Attrition.csv

    Returns
    -------
    pd.DataFrame
        Raw DataFrame – no transformations applied.

    Raises
    ------
    FileNotFoundError
        If the CSV is not found at `path`.
    """
    print(f"[data_processing] Loading raw data from: {path}")
    df = pd.read_csv(path, encoding=CSV_ENCODING)
    print(f"[data_processing] Loaded {len(df):,} rows × {len(df.columns)} columns")
    return df


# ── 2. Clean & encode ──────────────────────────────────────────────────────────

def clean_and_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and encode the raw DataFrame.

    Steps
    -----
    1. Drop constant columns and the ID column (no predictive value).
    2. Encode the target column: Yes → 1, No → 0.
    3. One-hot encode all remaining categorical (object) columns.
       ``drop_first=True`` avoids the dummy-variable trap.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame returned by ``load_raw()``.

    Returns
    -------
    pd.DataFrame
        Fully numeric DataFrame ready for modelling.
    """
    df = df.copy()

    # -- Step 1: drop uninformative columns ------------------------------------
    cols_to_drop = CONSTANT_COLUMNS + [ID_COLUMN]
    df = df.drop(columns=cols_to_drop)
    print(f"[data_processing] Dropped columns: {cols_to_drop}")

    # -- Step 2: encode target -------------------------------------------------
    df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})
    print(f"[data_processing] Target distribution:\n{df[TARGET].value_counts()}")

    # -- Step 3: one-hot encode categoricals -----------------------------------
    categorical_cols = df.select_dtypes(include="object").columns.tolist()
    print(f"[data_processing] One-hot encoding: {categorical_cols}")
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    print(f"[data_processing] Clean shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    return df


# ── 3. Save ────────────────────────────────────────────────────────────────────

def save_clean(df: pd.DataFrame, path: str = "data/attrition_clean.csv") -> None:
    """
    Save the cleaned DataFrame to CSV.

    Parameters
    ----------
    df   : pd.DataFrame
        Cleaned DataFrame returned by ``clean_and_encode()``.
    path : str, optional
        Destination path. Default: ``data/attrition_clean.csv``.
    """
    df.to_csv(path, index=False, encoding=CSV_ENCODING)
    print(f"[data_processing] Saved clean data to: {path}")


# ── 4. Split & scale ───────────────────────────────────────────────────────────

def split_and_scale(
    df: pd.DataFrame,
    target: str = TARGET,
    test_size: float = 0.2,
    scale: bool = False,
):
    """
    Split the cleaned DataFrame into train and test sets.

    The split is **stratified** on the target to preserve the ~16/84 class
    ratio in both sets – important given the class imbalance.

    Parameters
    ----------
    df        : pd.DataFrame
        Cleaned DataFrame returned by ``clean_and_encode()``.
    target    : str, optional
        Name of the target column. Default: ``"Attrition"``.
    test_size : float, optional
        Fraction of data held out for testing. Default: 0.2 (20 %).
    scale     : bool, optional
        If True, apply ``StandardScaler`` to features.
        Required for PCA and UMAP notebooks. Default: False.

    Returns
    -------
    Without scaling  → X_train, X_test, y_train, y_test
    With scaling     → X_train, X_test, y_train, y_test, scaler

    Notes
    -----
    The scaler is fitted on X_train only. X_test is transformed – never
    fitted – to prevent data leakage.
    """
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,        # preserve class ratio in both splits
    )

    print(
        f"[data_processing] Split: {len(X_train):,} train / {len(X_test):,} test "
        f"(stratified, test_size={test_size})"
    )

    if scale:
        scaler = StandardScaler()
        X_train = pd.DataFrame(
            scaler.fit_transform(X_train),
            columns=X_train.columns,
            index=X_train.index,
        )
        X_test = pd.DataFrame(
            scaler.transform(X_test),
            columns=X_test.columns,
            index=X_test.index,
        )
        print("[data_processing] Features scaled with StandardScaler")
        return X_train, X_test, y_train, y_test, scaler

    return X_train, X_test, y_train, y_test
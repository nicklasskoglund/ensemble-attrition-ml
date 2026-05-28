# Employee Attrition Prediction – Ensemble Methods

> **Course:** Ensemble Methods – Jensen Vocational College  
> **Dataset:** [IBM HR Analytics – Employee Attrition](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

---

## Overview

Replacing an employee costs a company up to **150% of their annual salary** — yet
most HR departments lack the tools to identify flight risks before it's too late.

This project builds a predictive model trained on IBM's HR dataset that flags
employees with a high probability of leaving. HR can then direct retention efforts
where they matter most, before resignations happen.

---

## Dataset

| Property        | Value                                   |
|-----------------|-----------------------------------------|
| Source          | Kaggle / IBM                            |
| Rows            | 1 470                                   |
| Features        | 35                                      |
| Target          | `Attrition` (Yes / No)                  |
| Class balance   | ~16 % Yes, ~84 % No                     |

### Getting the data

1. Go to the [Kaggle page](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)
2. Download `WA_Fn-UseC_-HR-Employee-Attrition.csv`
3. Place the file in the `data/` folder

> The raw file is **not committed** (see `.gitignore`).  
> Run `python main.py` to automatically generate `data/attrition_clean.csv`.

---

## Project Structure

```
ensemble_attrition/
├── data/
│   ├── .gitkeep                                    ← keeps folder in Git
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv       ← place here (not committed)
│
├── outputs/
│   ├── figures/                                    ← saved plots (.png)
│   └── results/                                    ← model results (.json)
│
├── notebooks/
│   ├── 01_EDA.ipynb                                ← Exploratory Data Analysis
│   ├── 02_decision_tree.ipynb                      ← Baseline: Decision Tree
│   ├── 03_random_forest.ipynb                      ← Random Forest + tuning
│   ├── 04_xgboost.ipynb                            ← XGBoost + tuning
│   ├── 05_pca_umap.ipynb                           ← Dimensionality reduction
│   └── 06_story.ipynb                              ← Data story / presentation
│
├── src/
│   ├── __init__.py                                 ← package definition
│   ├── data_processing.py                          ← loading, cleaning, splitting
│   ├── model_training.py                           ← training & hyperparameter tuning
│   └── evaluation.py                               ← metrics, plots, saving results
│
├── main.py                                         ← runs the full pipeline end-to-end
├── requirements.txt                                ← dependencies
└── README.md                                       ← this file
```

---

## Models

| Model             | Type       | Purpose                                    |
|-------------------|------------|--------------------------------------------|
| Decision Tree     | Baseline   | Simple reference point                     |
| Random Forest     | Ensemble   | Bagging – reduces variance                 |
| XGBoost           | Ensemble   | Boosting – reduces bias                    |
| RF / XGB + PCA    | Extra      | Linear dimensionality reduction            |
| RF / XGB + UMAP   | Extra      | Non-linear dimensionality reduction        |

---

## Evaluation Strategy

The dataset is **imbalanced** (84 % No, 16 % Yes). A model that always predicts
"No" achieves 84 % accuracy — without being useful.

We therefore evaluate using:

| Metric               | Reason                                              |
|----------------------|-----------------------------------------------------|
| **F1-score (Yes)**   | Balances Precision and Recall — primary metric      |
| **ROC-AUC**          | Measures how well the model ranks risk levels       |
| **Recall (Yes)**     | Most critical business metric — don't miss at-risk employees |
| ~~Accuracy~~         | Misleading under class imbalance — avoided          |

---

## Getting Started

```bash
# 1. Clone the repository
git clone https://github.com/<org>/ensemble_attrition.git
cd ensemble_attrition

# 2. Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

# 3. Install dependencies
pip install -r requirements.txt

# 4. Place raw data in data/ (see above)

# 5. Run the full pipeline
python main.py
```

---

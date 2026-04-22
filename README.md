# Financial Fraud Detection: Feature Engineering and Classification Pipeline

This project builds a machine learning pipeline to detect fraudulent financial transactions on the IEEE-CIS Fraud Detection dataset. The dataset is high-dimensional (434 features), heavily imbalanced (~3.5% fraud), and contains a mix of numerical, categorical, and 339 anonymized payment-network features (V1-V339). The goal is to correctly flag as many fraudulent transactions as possible while keeping false alarms at a manageable level.

**Dataset:** [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)  
**Target variable:** `isFraud` (0 = legitimate, 1 = fraudulent)  
**Class imbalance:** ~96.5% legitimate, ~3.5% fraud  
**Citation:** IEEE Computational Intelligence Society and Vesta Corporation. (2019). *IEEE-CIS Fraud Detection*. Kaggle. https://kaggle.com/competitions/ieee-fraud-detection

---

## Project Structure

```
Financial-Fraud-Detection/
├── notebooks/
│   ├── 01_Transaction_EDA.ipynb        - Exploratory data analysis
│   └── 02_Preprocessing.ipynb          - Feature engineering, preprocessing, and modeling
├── data/                               - Raw CSVs (not tracked in Git)
├── pyproject.toml                      - Python dependencies
└── README.md
```

---

## Methodology

### 1. Exploratory Data Analysis

The EDA notebook (`01_Transaction_EDA.ipynb`) covers all major feature groups: transaction amounts, temporal patterns, D-columns, email domains, card features, address fields, identity columns, and the anonymized V-columns. Key findings that shaped the modeling pipeline:

- `TransactionAmt` has a skewness of 14.37 and requires a log transformation before modeling.
- D-columns (D1-D15) drift over time as they count days since past events. A normalization step corrects for this.
- D9 is excluded entirely due to 87% missingness and a fundamentally different scale from the other D-columns.
- V-columns (V1-V339) are all missing to some degree. Sentinel fill is used instead of median imputation so that the model can distinguish "missing" from zero.
- card1, card2, card5, addr1, and addr2 have thousands of unique values. Frequency encoding is used to avoid one-hot explosion.

### 2. Feature Engineering

All feature engineering is done on the training set and applied to the test set without lookahead.

- **Temporal features:** `TransactionDT` is decoded to `Hour`, `Weekday`, and `Day` using the confirmed reference date of 2017-12-01.
- **D-column normalization:** Each D-column is converted to a fixed calendar reference by subtracting the current transaction day (`TransactionDT / 86400 - D_col`), removing temporal drift.
- **Email domain grouping:** Raw domains are mapped to Google, Yahoo, or Microsoft. Domains appearing fewer than 500 times are collapsed into "other". Missing values become their own "missing" category.
- **Frequency encoding:** card1, card2, card5, addr1, and addr2 are replaced with their frequency counts from the training set. A card appearing 50,000 times is likely a repeat customer; one appearing 3 times is suspicious.
- **Log transformation:** `TransactionAmt` is log-transformed using `np.log1p` to reduce skewness.

### 3. Preprocessing

- Columns with more than 50% missing values are dropped (V-columns are handled separately).
- V-columns are filled with `min_value - 1` (sentinel fill) so missingness remains visible to tree-based models.
- Numerical columns are imputed with the median; categorical columns with the most frequent value.
- Categorical columns are label-encoded. `get_dummies` is avoided to prevent column explosion from high-cardinality features.
- V-columns are scaled with MinMaxScaler and reduced from 339 dimensions to 30 principal components via PCA.

### 4. Modeling

Three models are trained and compared:

| Model | Key Setting |
|---|---|
| Decision Tree | `criterion='entropy'`, `max_depth=12`, `min_samples_leaf=50`, `class_weight='balanced'` |
| Random Forest | `n_estimators=100`, `class_weight='balanced'`, `n_jobs=-1` |
| XGBoost | `n_estimators=300`, `learning_rate=0.05`, `max_depth=6`, `scale_pos_weight` auto-set |

All models use `class_weight='balanced'` (or the XGBoost equivalent `scale_pos_weight`) to handle the severe class imbalance without oversampling.

---

## Results

Thresholds for Decision Tree and Random Forest are set to 0.4 (original). XGBoost uses an optimal threshold derived from Youden's J statistic. Full cross-validation and business cost analysis are included in `02_Preprocessing.ipynb`.

| Model | ROC-AUC | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|---|---|---|---|---|
| Decision Tree | ~0.88 | 0.12 | 0.82 | 0.21 |
| Random Forest | ~0.97 | 0.92 | 0.54 | 0.68 |
| XGBoost | ~0.98+ | see notebook | see notebook | see notebook |

**Key trade-off:** The Decision Tree catches more fraud (high recall) but flags many legitimate transactions (low precision). The Random Forest is more conservative. XGBoost generally achieves a better balance at its Youden-optimized threshold.

**Threshold justification:** Youden's J statistic (`J = TPR - FPR`) is computed for each model to find the threshold that maximizes sensitivity + specificity. This replaces the manually chosen 0.4.

**SHAP analysis:** SHAP (SHapley Additive exPlanations) is used on the XGBoost model to explain feature importance at both the individual and global level.

**Cross-validation:** 5-fold stratified cross-validation (ROC-AUC scoring) is used to validate that results are not a fluke of the particular train/test split.

---

## Business Context

In fraud detection, the two types of model error have very different costs:

- **False Negative (missed fraud):** The bank absorbs the chargeback. This is the more expensive error.
- **False Positive (false alarm):** A legitimate transaction is blocked or flagged for review. This inconveniences the customer.

The modeling decisions in this project (class weighting, threshold tuning, prioritizing recall) reflect this asymmetry. A cost analysis using illustrative values ($200 per missed fraud, $5 per false alarm) is included in the notebook.

---

## Setup

### Requirements

Python 3.10 or higher. Dependencies are managed with [uv](https://docs.astral.sh/uv/).

```bash
pip install uv
uv sync
```

Or install directly with pip:

```bash
pip install -r requirements.txt
```

### Data

Download the dataset from [Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place the four CSV files in a `data/` folder at the project root:

```
data/
├── train_transaction.csv
├── train_identity.csv
├── test_transaction.csv
└── test_identity.csv
```

### Running the notebooks

```bash
jupyter notebook
```

Open `01_Transaction_EDA.ipynb` first for the full EDA walkthrough, then `02_Preprocessing.ipynb` for the modeling pipeline.

---

## Limitations and Future Work

- The V-columns are fully anonymized, so their real-world meaning cannot be confirmed. The PCA representation loses some interpretability.
- Hyperparameters for the Decision Tree and Random Forest were chosen manually. A full GridSearchCV pass would likely improve both models.
- The cost analysis uses illustrative numbers. Real deployment would require calibrated chargeback rates and review costs.
- A real-time scoring pipeline (e.g. a FastAPI endpoint with a serialized model) would be needed for production use.
- SMOTE oversampling was explored in earlier iterations but replaced with `class_weight='balanced'`, which avoids creating synthetic data.

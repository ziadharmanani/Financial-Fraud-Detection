# Financial Fraud Detection: Feature Engineering and Classification Pipeline

This project builds a machine learning pipeline to detect fraudulent financial transactions on the IEEE-CIS Fraud Detection dataset. The dataset is high-dimensional (434 features), heavily imbalanced (~3.5% fraud), and contains a mix of numerical, categorical, and 339 anonymized payment-network features (V1-V339). The goal is to correctly flag as many fraudulent transactions as possible while keeping false alarms at a manageable level.

**Dataset:** [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)  
**Target variable:** `isFraud` (0 = legitimate, 1 = fraudulent)  
**Class imbalance:** ~96.5% legitimate, ~3.5% fraud  
**Citation:** IEEE Computational Intelligence Society and Vesta Corporation. (2019). *IEEE-CIS Fraud Detection*. Kaggle. https://kaggle.com/competitions/ieee-fraud-detection

## Project Structure

```
Financial-Fraud-Detection/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 02_feature_engineering_and_modelling.ipynb
├── data/ - Raw CSVs (not tracked in Git)
├── models/ (Not tracked)
├── utils/
│   └── helpers.py
├── requirements.txt
└── README.md
```

## Methodology

### 1. Exploratory Data Analysis

The EDA notebook (`01_eda.ipynb`) covers all major feature groups: transaction amounts, temporal patterns, D-columns, email domains, card features, address fields, identity columns, and the anonymized V-columns. Key findings that shaped the modelling pipeline:

- `TransactionAmt` has a skewness of 14.37 and requires a log transformation before modelling.
- D-columns (D1-D15) drift over time as they count days since past events. A normalization step corrects for this.
- D9 is excluded entirely due to 87% missingness and a fundamentally different scale from the other D-columns.
- V-columns (V1-V339) are all missing to some degree and exhibit grouped missingness. Blocks of columns from the same data source are missing together.
- card1, card2, card5, addr1, and addr2 have thousands of unique values. Frequency encoding is used to avoid one-hot explosion.

### 2. Feature Engineering

All feature engineering is done on the training set and applied to the test set without lookahead.

- **Temporal features:** `TransactionDT` is decoded to `Hour`, `Weekday`, and `Day` using the confirmed reference date of 2017-12-01.
- **D-column normalization:** Each D-column is converted to a fixed calendar reference by subtracting the current transaction day (`TransactionDT / 86400 - D_col`), removing temporal drift.
- **Email domain grouping:** Raw domains are mapped to Google, Yahoo, or Microsoft. Domains appearing fewer than 500 times are collapsed into "other". Missing values become their own "missing" category.
- **Categorical missing values:** All categorical NaN values are filled with an explicit `'missing'` category rather than mode imputation. This ensures the model can distinguish "data was unavailable" from any observed value, which is particularly important for columns like `DeviceType` where 76% of rows lack identity data.
- **Log transformation:** `TransactionAmt` is log-transformed using `np.log1p` to reduce skewness.

### 3. Preprocessing

The preprocessing pipeline handles different feature groups based on their characteristics and what downstream transformations they feed into.

- **Column missingness:** Non-V columns above 80% missing are dropped. This retains ID columns in the 50-80% range that were previously discarded.
- **V-column missingness groups:** V-columns share grouped missingness patterns (e.g., V1-V11 are always missing together). We identify these groups and create one binary flag per group, then impute the V-values with the column median. The flags capture "this data source was unavailable" while median fill provides neutral values for PCA.
- **D-column sentinel fill:** Normalized D-columns represent calendar reference dates. Missing means the reference event never happened, so a sentinel value (`min - 1`) is used instead of the median.
- **Numerical missingness flags:** Columns with >2% missingness get a binary `_missing` indicator before median imputation.
- **Split-before-encoding:** The train/test split happens before any frequency encoding, label encoding, or PCA fitting. This prevents information leakage from validation rows into encoding statistics.
- **Frequency encoding:** card1, card2, card5, addr1, and addr2 are replaced with frequency counts computed from the training set only. Unseen test values receive frequency 0.
- **Label encoding:** Categorical columns are label-encoded using training data only. Unseen test categories are mapped to a dedicated "unknown" integer.
- **PCA on V-columns:** V-columns are scaled with MinMaxScaler and reduced via PCA. The number of components is selected to explain 95% of variance rather than hardcoded.

### 4. Modelling

Three models are trained and compared, all using `class_weight='balanced'` to handle the severe class imbalance without oversampling:

| Model | Key Settings |
|---|---|
| Decision Tree | `criterion='entropy'`, `max_depth=12`, `min_samples_leaf=50` |
| Random Forest | `n_estimators=100`, `max_depth=15`, `n_jobs=-1` |
| LightGBM (Tuned) | Optuna-tuned: `num_leaves=132`, `max_depth=11`, `learning_rate=0.044`, `n_estimators=1000` with early stopping |

LightGBM hyperparameters were tuned using Optuna (30 trials, 3-fold stratified CV, Bayesian search). The best configuration is saved to `models/lgbm_best_params.json` to avoid re-running the search.

## Results

All models use an optimal threshold derived from the maximum F1-score on the precision-recall curve. This is more appropriate than Youden's J for severely imbalanced datasets, because Youden's J can sacrifice thousands of false positives to catch a few extra frauds. The enormous True Negative count masks the cost.

| Model | ROC-AUC | Precision (Fraud) | Recall (Fraud) | F1 (Fraud) |
|---|---|---|---|---|
| Decision Tree | 0.8783 | 0.62 | 0.45 | 0.52 |
| Random Forest | 0.9411 | 0.67 | 0.56 | 0.61 |
| LightGBM (Tuned) | 0.9598 | 0.87 | 0.72 | 0.79 |

The tuned LightGBM model catches 72% of all fraud while being correct 87% of the time when it flags a transaction.

## Business Context

In fraud detection, the two types of model error have very different costs:

- **False Negative (missed fraud):** The bank absorbs the chargeback. This is the more expensive error.
- **False Positive (false alarm):** A legitimate transaction is blocked or flagged for review. This inconveniences the customer.

The modelling decisions in this project (class weighting, threshold tuning, prioritizing recall) reflect this asymmetry. A cost analysis using illustrative values ($200 per missed fraud, $5 per false alarm) is included in the notebook.

## Setup

### Requirements

Python 3.10 or higher.

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

**Note:** The test identity file uses hyphens in column names (`id-01`) while the train file uses underscores (`id_01`). The notebook normalizes this automatically on load.

### Running the notebooks

```bash
jupyter notebook
```

Open `01_eda.ipynb` first for the full EDA walkthrough, then `02_feature_engineering_and_modelling.ipynb` for the modelling pipeline.

## Privacy and Compliance Considerations

This project uses the IEEE-CIS dataset, in which all cardholder identifiers and payment-network features (V1-V339) have been anonymized by Vesta Corporation prior to release. No personally identifiable information (PII) is present in the raw data.

In a production deployment at a financial institution, several additional considerations would apply. Transaction data containing cardholder details is subject to **PCI-DSS** (Payment Card Industry Data Security Standard), which governs how card data is stored, transmitted, and processed. In jurisdictions where cardholders are EU residents, **GDPR** would impose further constraints on data retention and the right to explanation. In Canada, **PIPEDA** applies similar principles. Any real scoring pipeline would need to operate within these frameworks, including data minimization, access controls, audit logging, and the ability to provide human-readable explanations for adverse decisions (which is directly addressed by the SHAP analysis in this project).

---

## Limitations and Future Work

- The V-columns are fully anonymized, so their real-world meaning cannot be confirmed. The PCA representation retains 95% of variance but loses some interpretability.
- `TransactionDT` is dropped after temporal feature extraction. Retaining it could enable velocity features (transactions per hour per card) and recency features that winning competition solutions relied on.
- UID-based aggregation features (e.g., combining card1 + addr1 + D1 to create a client identifier and computing per-client statistics) were the single biggest technique in winning solutions and are not yet implemented.
- Ensemble methods (stacking or blending the three models) could further improve performance.
- The cost analysis uses illustrative numbers. Real deployment would require calibrated chargeback rates and review costs.
- A real-time scoring pipeline (a FastAPI endpoint with a serialized model) would be needed for production use.

import pandas as pd
import numpy as np
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder  # Added for split-safe label encoding

def SummarizeTable(df):
    """
    Generates a summary table for a given DataFrame including dtypes, 
    missing values, unique values, entropy, and the first three rows.
    """
    print(f"Dataset Shape: {df.shape}")
    summary = pd.DataFrame(df.dtypes, columns=['dtypes'])
    summary = summary.reset_index()
    summary['Name'] = summary['index']
    summary = summary[['Name', 'dtypes']]
    summary['Missing'] = df.isnull().sum().values
    summary['Uniques'] = df.nunique().values
    
    # Using iloc instead of loc to avoid KeyErrors if the index isn't standard
    summary['First Value'] = df.iloc[0].values if len(df) > 0 else np.nan
    summary['Second Value'] = df.iloc[1].values if len(df) > 1 else np.nan
    summary['Third Value'] = df.iloc[2].values if len(df) > 2 else np.nan

    for name in summary['Name'].value_counts().index:
        summary.loc[summary['Name'] == name, 'Entropy'] = round(
            stats.entropy(df[name].value_counts(normalize=True), base=2), 2
        ) 

    return summary

def CalcOutliers(df_num): 
    """
    Calculates and prints the number of outliers in a numeric array/series
    using the 3 Standard Deviations rule.
    """
    # calculating mean and std of the array
    data_mean, data_std = np.mean(df_num), np.std(df_num)

    # setting the cut line to both higher and lower values
    cut = data_std * 3

    # Calculating the higher and lower cut values
    lower, upper = data_mean - cut, data_mean + cut

    # creating an array of lower, higher and total outlier values 
    outliers_lower = [x for x in df_num if x < lower]
    outliers_higher = [x for x in df_num if x > upper]
    outliers_total = [x for x in df_num if x < lower or x > upper]

    # array without outlier values
    outliers_removed = [x for x in df_num if x >= lower and x <= upper]
    
    print('Identified lowest outliers: %d' % len(outliers_lower)) 
    print('Identified upper outliers: %d' % len(outliers_higher)) 
    print('Total outlier observations: %d' % len(outliers_total)) 
    print('Non-outlier observations: %d' % len(outliers_removed)) 
    
    # Corrected percentage calculation (outliers / total observations)
    total_obs = len(df_num)
    if total_obs > 0:
        print("Total percentage of Outliers: ", round((len(outliers_total) / total_obs) * 100, 4), "%")
    else:
        print("Total percentage of Outliers: 0.0 %")
    
    return

def PCAFitTransform(df_train, df_test, cols, n_components, prefix='PCA_', rand_seed=4):
    """
    Fits PCA on training data, transforms both train and test data, 
    and concatenates the new PCA components with the original DataFrames 
    (dropping the original columns used for PCA).
    """
    pca = PCA(n_components=n_components, random_state=rand_seed)

    # Fit on train, transform both
    train_pca = pca.fit_transform(df_train[cols])
    test_pca = pca.transform(df_test[cols])

    # Convert to DataFrame
    train_pca_df = pd.DataFrame(train_pca, columns=[f"{prefix}{i}" for i in range(n_components)])
    test_pca_df = pd.DataFrame(test_pca, columns=[f"{prefix}{i}" for i in range(n_components)])

    # Drop original columns and reset index for clean concatenation
    df_train = df_train.drop(columns=cols).reset_index(drop=True)
    df_test = df_test.drop(columns=cols).reset_index(drop=True)

    # Concatenate PCA features with the remaining original features
    df_train_final = pd.concat([df_train, train_pca_df], axis=1)
    df_test_final = pd.concat([df_test, test_pca_df], axis=1)

    return df_train_final, df_test_final


# ──────────────────────────────────────────────────────────────────────
# New utility functions added during preprocessing rework.
# These address specific issues identified in the original pipeline:
#   1. V-column missingness was handled with sentinel fill before PCA,
#      which injected artificial variance. Now we separate missingness
#      flags from the values that PCA consumes.
#   2. Frequency and label encoding were fitted on the full dataset
#      (including future validation rows). Now they fit on training
#      data only, matching a real deployment setting.
# ──────────────────────────────────────────────────────────────────────


def detect_v_missingness_groups(df, v_cols):
    """
    Identifies groups of V-columns that share identical missingness patterns.
    V-columns aren't 339 independent features — they come in blocks from
    different data sources. If V1 is missing for a row, V2-V11 are too.

    Returns
    -------
    groups : dict
        Mapping of group name -> list of column names in that group.
    flags  : pd.DataFrame
        One binary column per group (1 = entire group missing, 0 = present).
    """
    null_mask = df[v_cols].isnull()

    groups = {}
    assigned = set()
    group_id = 0

    for col in v_cols:
        if col in assigned:
            continue

        # Find every column whose missingness pattern matches this one exactly
        pattern = null_mask[col]
        matching = [c for c in v_cols if c not in assigned and (null_mask[c] == pattern).all()]

        group_name = f'v_missing_group_{group_id}'
        groups[group_name] = matching
        assigned.update(matching)
        group_id += 1

    # Build a compact flag DataFrame — one column per group
    flags = pd.DataFrame(index=df.index)
    for group_name, cols in groups.items():
        # Only need to check the first column since all share the same pattern
        flags[group_name] = df[cols[0]].isnull().astype(int)

    return groups, flags


def split_safe_frequency_encode(X_train, X_test, cols):
    """
    Frequency-encodes high-cardinality columns using counts derived
    exclusively from the training set. This avoids the subtle leakage
    that occurs when frequencies are computed on the full dataset
    (including rows that later become the validation set).

    Test-set values not seen during training receive frequency 0.
    """
    for col in cols:
        # Compute frequencies from training data only
        freq_map = X_train[col].value_counts()
        X_train.loc[:, col] = X_train[col].map(freq_map)
        X_test.loc[:, col] = X_test[col].map(freq_map).fillna(0)

    return X_train, X_test


def split_safe_label_encode(X_train, X_test, cols):
    """
    Label-encodes categorical columns using only training data.
    Unseen categories in the test set are mapped to a dedicated
    'unknown' integer (max_label + 1), which mirrors how a production
    pipeline would handle a new category it has never seen before.

    Returns the modified DataFrames and a dict of fitted encoders
    (useful if you want to inspect or reuse them later).
    """
    encoders = {}

    for col in cols:
        le = LabelEncoder()

        # Fit only on training data
        le.fit(X_train[col].astype(str))
        encoders[col] = le

        # Transform training data (all values are guaranteed known)
        X_train.loc[:, col] = le.transform(X_train[col].astype(str)).astype(int)

        # Transform test data, routing unseen categories to max + 1.
        # .apply() returns object dtype, so we explicitly cast to int
        # to keep LightGBM (and other frameworks) happy.
        known_classes = set(le.classes_)
        unknown_label = len(le.classes_)
        X_test.loc[:, col] = X_test[col].astype(str).apply(
            lambda x, k=known_classes, u=unknown_label, enc=le:
                enc.transform([x])[0] if x in k else u
        ).astype(int)

    return X_train, X_test, encoders
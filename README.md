# Financial Fraud Detection: Feature Engineering and Classification Pipeline
This project focuses on building a robust machine learning pipeline to detect fraudulent financial transactions. Fraud detection datasets are notoriously imbalanced, skewed and high-dimensional. 
Feature selection, missing value imputation, preprocessing of highly correlated variables, and the optimization of tree-based classifiers (Decision Trees and Random Forests).

The data is sourced from the **IEEE-CIS Fraud Detection** dataset, provided by the IEEE Computational Intelligence Society and Vesta Corporation. It contains transactions with a wide array of numerical, categorical, and anonymized "V-features" engineered by the payment network.

* **Source:** [IEEE-CIS Fraud Detection on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data)
* **Target Variable:** `isFraud` (Binary: 0 for legitimate, 1 for fraudulent)
* **Class Imbalance:** Extremely high (~96.5% legitimate, ~3.5% fraud)
* **Citation:** IEEE Computational Intelligence Society & Vesta Corporation. (2019). *IEEE-CIS Fraud Detection*. Kaggle. https://kaggle.com/competitions/ieee-fraud-detection

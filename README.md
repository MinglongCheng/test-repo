# test-repo

# Tabular Binary Classification ML & SHAP Consistency Pipeline

A complete, end‑to‑end machine learning pipeline for **tabular binary classification** on a medical-style dataset, including:

- data preparation and feature engineering  
- multiple machine learning models with hyperparameter tuning  
- rich model evaluation and visualization  
- SHAP-based model explainability  
- **cross-model SHAP direction consistency analysis** (custom extension)

All code is implemented in Python and contains **detailed in-line comments in Chinese**, making the project suitable as a reference template for similar tabular ML tasks.

> Note: The dataset in this repository is a small illustrative health-related dataset used purely for demonstration and education.  
> The workflow is generic and can be adapted to other binary classification problems.

---

## 1. What this project does

The code implements a full workflow from raw CSV to interpretable models:

1. **Data understanding & cleaning**
2. **Train/test split & baseline balance check**
3. **Feature engineering & variable selection**
4. **Model training & hyperparameter tuning**
5. **Model evaluation on validation and test sets**
6. **SHAP-based explainability for each model**
7. **Cross-model SHAP direction consistency analysis with radar plots**

The focus is on the **machine learning process and interpretability**, not on a specific disease.

---

## 2. Pipeline overview

### 2.1 Data understanding & exploration

Main capabilities:

- Load raw CSV from `data/raw/`.
- Inspect structure and basic statistics (`head`, `info`, missing counts).
- Summaries of the binary outcome distribution (frequency & proportion).
- Visual exploration of representative continuous features (e.g. histograms & boxplots).

Example figure (you can enable saving in code):

```python
plt.savefig("results/performance/example_feature_distribution.png",
            dpi=300, bbox_inches="tight")
```

### 2.2 Data cleaning & missing value imputation

Main steps:

- Detect outliers in a continuous feature using IQR (interquartile range).
- Adjust clearly erroneous outliers based on simple rules.
- Use MissForest (random-forest-based imputation) to fill missing values in a continuous variable.
- Preserve integer type for appropriate variables after imputation.
- Save the cleaned & imputed dataset to:
    - data/processed/diabetes_imputed.csv (can be replaced by your own tabular dataset name).

---

### 2.3 Train/test split & baseline comparison

Main steps:

- Stratified split into training set and test set (e.g. 70/30, preserving outcome ratio).
- Output:
    `train_data_notscaled.csv`
    `test_data_notscaled.csv`
- Add a group variable (train_set / test_set) and use TableOne to produce a descriptive table comparing baseline characteristics between train and test.
- Save baseline balance table to:
    - `results/tables/varbalance_table.csv`

This makes it easy to check whether the split is reasonably balanced.

---

### 2.4 Feature engineering & variable selection

#### Standardization

- Continuous variables (e.g. age, laboratory measurements, indices) are standardized using StandardScaler.
- Fit the scaler on the training set, then apply to the test set.
- Output:
    - `train_data_scaled.csv`
    - `test_data_scaled.csv`

#### Two complementary feature selection strategies

##### 1.Univariable + multivariable logistic regression

    - Fit univariable logistic models for each candidate predictor.
    - Select variables with significant p-values.
    - Fit a multivariable logistic regression on the selected set.
    - Save the multivariable regression results to:
        - `results/tables/results_mulvariable_df.csv`
    - Save the final selected variable list to:
        - `models/significant_vars.pkl`

##### 2.LASSO regression

    - Fit a Lasso model on all candidate features.
    - Select features with non-zero coefficients.
    - Provides an alternative variable selection pathway to compare against logistic-based selection.

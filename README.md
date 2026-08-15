# Bank Marketing – Term Deposit Subscription Prediction

## a. Problem Statement

Portuguese banks run direct marketing campaigns (phone calls) to convince clients to
subscribe to a **term deposit**. Contacting every client is expensive, so the bank needs a
way to identify, in advance, which clients are most likely to subscribe.

The goal of this project is to build and compare supervised **binary classification** models
that predict whether a client will subscribe to a term deposit (target variable `y`,
values `yes`/`no`). A good model lets the bank focus its calls on the clients with the
highest probability of subscribing, improving campaign efficiency.

## b. Dataset Description

- **Source:** Bank Marketing Data Set (Moro et al., 2011) – Portuguese banking institution direct marketing campaigns (May 2008 – November 2010).
- **Files used:**
  - `bank-full_training_data.csv` – **training data**, 45,211 records (all examples).
  - `bank_testdata.csv` – **test data**, 4,521 records (a 10% random sample of the full set).
  - Both files are **semicolon (`;`) separated**.
- **Attributes:** 16 input features + 1 output/target = 17 columns.
- **Target variable:** `y` – has the client subscribed a term deposit? (`yes` / `no`).
- **Class balance:** highly **imbalanced** – about 88% `no` vs 12% `yes` in the training set.

### Feature Overview

| Type | Features |
|------|----------|
| Numeric (7) | `age`, `balance`, `day`, `duration`, `campaign`, `pdays`, `previous` |
| Categorical (9) | `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome` |

### Preprocessing Applied

- Encoded the target `y` as `1` (yes) / `0` (no).
- **One-hot encoding** of the 9 categorical features, with train/test columns aligned (→ 51 features).
- Missing-value imputation and **z-score outlier capping** on numeric features (fit on training data).
- **Power transformation** (Yeo-Johnson) to reduce skew, then **standardization** (zero mean / unit variance).
- Class imbalance handled with `class_weight='balanced'` where supported.

## c. GitHub Repository Link

https://github.com/2025ac05780-crypto/Bank_Term_Deposit_Subscription_Prediction.git
(2025ac05780-crypto/Bank_Term_Deposit_Subscription_Prediction)
###  Repository Structure
```
Bank_Term_Deposit_Subscription_Prediction/.    
├── 2025ac05780_assignment2_ML.ipynb     # Main notebook: preprocessing, 5 models, 6 metrics
├── app.py                               # Streamlit interactive frontend (deployed app)
├── bank-full_training_data.csv          # Training dataset (45,211 records)
├── bank-names.txt                       # Dataset attribute description
├── bank_testdata.csv                    # Test dataset (4,521 records)
├── README.md                            # This file contains summary
└── requirements.txt                     # Python dependencies for reproducible runs / deployment

0 directories, 7 files
```
## d. Models Used

The following five classification models were trained on the **same** preprocessed dataset
(`bank-full_training_data.csv` for training, `bank_testdata.csv` for testing):

1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor Classifier
4. Naive Bayes Classifier (Gaussian)
5. Ensemble Model – Random Forest

For each model, six evaluation metrics were computed on the test set: **Accuracy, AUC Score,
Precision, Recall, F1 Score, and Matthews Correlation Coefficient (MCC)**.

### Comparison Table

| ML Model                | Accuracy | AUC Score | Precision | Recall | F1 Score | MCC Score |
| ----------------------- | :------: | :-------: | :-------: | :----: | :------: | :-------: |
| Logistic Regression     |  0.7945  |   0.9010  |   0.3445  | 0.8676 |  0.4932  |   0.4591  |
| Decision Tree           |  0.9821  |   0.9365  |   0.9641  | 0.8772 |  0.9186  |   0.9098  |
| K-Nearest Neighbour     |  0.9115  |   0.9351  |   0.7065  | 0.3973 |  0.5086  |   0.4874  |
| Naive Bayes (Gaussian)  |  0.8350  |   0.7669  |   0.3323  | 0.4280 |  0.3742  |   0.2838  |
| **Random Forest**       |**0.9847**| **0.9992**| **0.9934**| 0.8733 |**0.9295**| **0.9234**|

### Observations

| ML Model Name                   | Observation about model performance |
| ------------------------------- | ----------------------------------- |
| **Logistic Regression**         | Lowest accuracy (0.79) because `class_weight='balanced'` pushes it to flag many potential subscribers, giving high **recall (0.87)** but very low **precision (0.34)** – many false positives. Still a strong ranking model (AUC 0.90). Good when catching subscribers matters more than precision. |
| **Decision Tree**               | Excellent, well-rounded performance (accuracy 0.98, F1 0.92, MCC 0.91). Balances precision and recall well and is easy to interpret, though a single tree can risk overfitting. |
| **kNN**                         | High accuracy (0.91) but weak **recall (0.40)** – it misses more than half of the actual subscribers, so its F1 (0.51) and MCC (0.49) are modest. Sensitive to the imbalanced minority class and to feature scaling. |
| **Naive Bayes**                 | Weakest overall (accuracy 0.84, F1 0.37, MCC 0.28). Its feature-independence assumption does not hold for this dataset, hurting both precision and recall. Fast but not competitive here. |
| **Random Forest (Ensemble)**    | **Best model.** Highest accuracy (0.98), near-perfect **AUC (0.999)**, highest precision (0.99), F1 (0.93) and MCC (0.92). The ensemble of trees generalizes far better than a single tree and handles the mix of encoded categorical and numeric features robustly. |
| **Overall Winner for dataset?** | **Random Forest** – it leads on nearly every metric (Accuracy, AUC, Precision, F1, MCC) and provides the most reliable, well-balanced predictions on the imbalanced Bank Marketing data. |

## e. Live Streamlit App

**Live App Link:** > https://banktermdepositsubscriptionprediction-m6aemk8httkaik5cfdqcwx.streamlit.app

The app ([app.py](app.py)) provides:
- A **Predict** tab: interactive form to enter client details and choose a model, returning the subscription probability.
- A **Model Comparison** tab: the evaluation-metrics table and bar chart for all five models.

### How to Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

**Missing dependencies are the most common cause of deployment failure** – keep `requirements.txt` complete and up to date.


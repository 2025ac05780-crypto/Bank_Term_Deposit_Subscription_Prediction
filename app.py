"""
Bank Marketing - Term Deposit Subscription Prediction
Interactive Streamlit frontend.

Deploy on Streamlit Community Cloud:
  - Main file path: app.py
  - Requires: requirements.txt, bank-full_training_data.csv, bank_testdata.csv
"""

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.preprocessing import PowerTransformer, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)

TARGET = "y"
TRAIN_FILE = "bank-full_training_data.csv"
TEST_FILE = "bank_testdata.csv"

st.set_page_config(page_title="Bank Marketing Predictor", page_icon="🏦", layout="wide")


@st.cache_data(show_spinner=False)
def load_raw():
    """Load the semicolon-separated train/test CSV files."""
    train = pd.read_csv(TRAIN_FILE, sep=";")
    test = pd.read_csv(TEST_FILE, sep=";")
    return train, test


@st.cache_resource(show_spinner="Preparing data and training models...")
def build_pipeline():
    """Preprocess the data, train all five models and return everything the UI needs."""
    train_raw, test_raw = load_raw()
    train, test = train_raw.copy(), test_raw.copy()

    # Encode the binary target: yes -> 1, no -> 0
    train[TARGET] = train[TARGET].map({"yes": 1, "no": 0})
    test[TARGET] = test[TARGET].map({"yes": 1, "no": 0})

    # Numeric feature columns; everything else (except the target) is categorical.
    # Selecting by "not numeric" is robust across pandas versions (object vs str dtype).
    numeric_cols = [c for c in train.select_dtypes(include=[np.number]).columns if c != TARGET]
    categorical_cols = [c for c in train.columns if c != TARGET and c not in numeric_cols]

    # One-hot encode categorical features and align train/test columns
    train_enc = pd.get_dummies(train, columns=categorical_cols)
    test_enc = pd.get_dummies(test, columns=categorical_cols)
    train_enc, test_enc = train_enc.align(test_enc, join="left", axis=1, fill_value=0)

    # Impute missing numeric values with the training mean
    train_means = train_enc[numeric_cols].mean()
    train_enc[numeric_cols] = train_enc[numeric_cols].fillna(train_means)
    test_enc[numeric_cols] = test_enc[numeric_cols].fillna(train_means)

    # Cap outliers using the z-score method (fit on training data)
    z = np.abs((train_enc[numeric_cols] - train_enc[numeric_cols].mean()) / train_enc[numeric_cols].std())
    train_enc[numeric_cols] = train_enc[numeric_cols].mask(z > 3, np.nan)
    train_enc[numeric_cols] = train_enc[numeric_cols].fillna(train_enc[numeric_cols].mean())

    # Power transform + standardize the numeric features (fit on train)
    pt = PowerTransformer()
    scaler = StandardScaler()
    train_enc[numeric_cols] = scaler.fit_transform(pt.fit_transform(train_enc[numeric_cols]))
    test_enc[numeric_cols] = scaler.transform(pt.transform(test_enc[numeric_cols]))

    feature_cols = [c for c in train_enc.columns if c != TARGET]
    X_train, y_train = train_enc[feature_cols], train_enc[TARGET].astype(int)
    X_test, y_test = test_enc[feature_cols], test_enc[TARGET].astype(int)

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, solver="saga",
                                                   class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(class_weight="balanced", random_state=42),
        "K-Nearest Neighbour": KNeighborsClassifier(n_neighbors=5),
        "Naive Bayes (Gaussian)": GaussianNB(),
        "Random Forest": RandomForestClassifier(n_estimators=100, class_weight="balanced",
                                                 random_state=42),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_score = model.predict_proba(X_test)[:, 1]
        rows.append({
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "AUC Score": roc_auc_score(y_test, y_score),
            "Precision": precision_score(y_test, y_pred, zero_division=0),
            "Recall": recall_score(y_test, y_pred, zero_division=0),
            "F1 Score": f1_score(y_test, y_pred, zero_division=0),
            "MCC Score": matthews_corrcoef(y_test, y_pred),
        })
    metrics_df = pd.DataFrame(rows).set_index("Model").round(4)

    return {
        "models": models,
        "metrics": metrics_df,
        "feature_cols": feature_cols,
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols,
        "pt": pt,
        "scaler": scaler,
        "raw_train": train_raw,
    }


def prepare_input(record, ctx):
    """Turn a single raw client record into the model-ready feature vector."""
    row = pd.DataFrame([record])
    row_enc = pd.get_dummies(row, columns=ctx["categorical_cols"])
    row_enc = row_enc.reindex(columns=ctx["feature_cols"], fill_value=0)
    row_enc[ctx["numeric_cols"]] = ctx["scaler"].transform(
        ctx["pt"].transform(row_enc[ctx["numeric_cols"]])
    )
    return row_enc


def build_observations(metrics_df):
    """Build the 'observations' table (attached format) from the computed metrics."""
    best = metrics_df.mean(axis=1).idxmax()
    rows = []
    for model, m in metrics_df.iterrows():
        obs = (f"Accuracy {m['Accuracy']:.2f}, AUC {m['AUC Score']:.2f}, "
               f"Precision {m['Precision']:.2f}, Recall {m['Recall']:.2f}, "
               f"F1 {m['F1 Score']:.2f}, MCC {m['MCC Score']:.2f}. ")
        if m['F1 Score'] >= 0.85 and m['MCC Score'] >= 0.85:
            obs += "Strong, well-balanced performance across every metric."
        elif m['Recall'] >= 0.80 and m['Precision'] < 0.5:
            obs += "High recall but low precision (many false positives) due to class balancing."
        elif m['Recall'] < 0.5:
            obs += "High accuracy but weak recall, so it misses many actual subscribers."
        else:
            obs += "Moderate performance; not the strongest on this dataset."
        if model == best:
            obs = "**Best model.** " + obs
        rows.append({"ML Model Name": model, "Observation about model performance": obs})

    rows.append({
        "ML Model Name": "Overall Winner for your dataset?",
        "Observation about model performance":
            f"**{best}** – highest average score across all six evaluation metrics, giving the "
            f"most reliable and well-balanced predictions on the imbalanced Bank Marketing dataset.",
    })
    return pd.DataFrame(rows).set_index("ML Model Name")


# ------------------------------------------------------------------ UI
st.title("🏦 Bank Marketing – Term Deposit Subscription Predictor")
st.caption("Predict whether a client will subscribe to a term deposit, using models trained on the Bank Marketing dataset.")

ctx = build_pipeline()
raw_train = ctx["raw_train"]

tab_predict, tab_metrics = st.tabs(["🔮 Predict", "📊 Model Comparison"])

with tab_metrics:
    st.subheader("Evaluation metrics for all 5 models (test set)")
    st.dataframe(ctx["metrics"], use_container_width=True)
    st.bar_chart(ctx["metrics"])
    best = ctx["metrics"].mean(axis=1).idxmax()
    st.success(f"Overall best model by average score: **{best}**")

    st.subheader("Observations on the performance of each model on the chosen dataset")
    st.table(build_observations(ctx["metrics"]))

with tab_predict:
    st.subheader("Enter client details")
    model_name = st.selectbox("Choose a model", list(ctx["models"].keys()), index=4)

    col1, col2, col3 = st.columns(3)
    record = {}
    with col1:
        record["age"] = st.slider("Age", 18, 95, 40)
        record["job"] = st.selectbox("Job", sorted(raw_train["job"].unique()))
        record["marital"] = st.selectbox("Marital status", sorted(raw_train["marital"].unique()))
        record["education"] = st.selectbox("Education", sorted(raw_train["education"].unique()))
        record["default"] = st.selectbox("Credit in default?", sorted(raw_train["default"].unique()))
    with col2:
        record["balance"] = st.number_input("Yearly balance (€)", -8000, 110000, 1000)
        record["housing"] = st.selectbox("Housing loan?", sorted(raw_train["housing"].unique()))
        record["loan"] = st.selectbox("Personal loan?", sorted(raw_train["loan"].unique()))
        record["contact"] = st.selectbox("Contact type", sorted(raw_train["contact"].unique()))
        record["month"] = st.selectbox("Last contact month", sorted(raw_train["month"].unique()))
    with col3:
        record["day"] = st.slider("Last contact day", 1, 31, 15)
        record["duration"] = st.number_input("Last contact duration (s)", 0, 5000, 200)
        record["campaign"] = st.number_input("Contacts this campaign", 1, 60, 1)
        record["pdays"] = st.number_input("Days since last contact (-1 = never)", -1, 900, -1)
        record["previous"] = st.number_input("Previous contacts", 0, 300, 0)
        record["poutcome"] = st.selectbox("Previous outcome", sorted(raw_train["poutcome"].unique()))

    if st.button("Predict", type="primary"):
        model = ctx["models"][model_name]
        X_one = prepare_input(record, ctx)
        proba = float(model.predict_proba(X_one)[:, 1][0])
        pred = int(proba >= 0.5)
        if pred == 1:
            st.success(f"✅ Likely to SUBSCRIBE — probability {proba:.1%}  (model: {model_name})")
        else:
            st.warning(f"❌ Unlikely to subscribe — probability {proba:.1%}  (model: {model_name})")

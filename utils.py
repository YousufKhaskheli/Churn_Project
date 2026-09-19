from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

BASE_DIR = Path(__file__).parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn.csv"
MODEL_PATH = BASE_DIR / "models" / "churn_pipeline.joblib"
RESULTS_PATH = BASE_DIR / "models" / "results.csv"
VISUALS_DIR = BASE_DIR / "visuals"

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CHAMPION = "XGBoost (tuned)"
MODEL_NAMES = ["Logistic Regression", "Decision Tree", "Random Forest",
               "KNN", "SVM", "XGBoost"]


@st.cache_data
def load_data():
    """Same cleaning as the notebook."""
    df = pd.read_csv(DATA_PATH)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df = df.drop(columns="customerID")
    df["Churn"] = (df["Churn"] == "Yes").astype(int)
    return df


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_results():
    return pd.read_csv(RESULTS_PATH)


def build_preprocess(X):
    cat_cols = [c for c in X.columns if c not in NUM_COLS + ["SeniorCitizen"]]
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
    ], remainder="passthrough")


def get_models(y_train):
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    return {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(class_weight="balanced", random_state=42, n_jobs=-1),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True, class_weight="balanced", random_state=42),
        "XGBoost": XGBClassifier(eval_metric="logloss", scale_pos_weight=scale_pos, random_state=42),
    }


@st.cache_resource(show_spinner=False)
def train_model(name):
    """Train one baseline model on the same split as the notebook."""
    df = load_data()
    X, y = df.drop(columns="Churn"), df["Churn"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)
    pipe = Pipeline([("prep", build_preprocess(X)),
                     ("model", get_models(y_train)[name])])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {"Accuracy": accuracy_score(y_test, pred),
               "Precision": precision_score(y_test, pred),
               "Recall": recall_score(y_test, pred),
               "F1": f1_score(y_test, pred),
               "ROC-AUC": roc_auc_score(y_test, proba)}
    return {"metrics": metrics, "y_test": y_test, "pred": pred}
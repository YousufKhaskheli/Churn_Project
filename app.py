import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.metrics import confusion_matrix

import utils

st.set_page_config(page_title="Customer Churn Dashboard", page_icon="📉", layout="wide")

df = utils.load_data()
FEATURES = df.drop(columns="Churn").columns.tolist()
CAT_COLS = [c for c in FEATURES if c not in utils.NUM_COLS + ["SeniorCitizen"]]
METRICS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]


def home():
    st.title("Customer Churn Prediction & Model Comparison")
    st.markdown(
        "Acquiring a new customer costs far more than keeping an existing one. "
        "This project predicts which telecom customers are likely to leave, so a "
        "retention team can offer loyalty incentives or better plans before they go."
    )
    c1, c2, c3 = st.columns(3)
    c1.metric("Customers", f"{len(df):,}")
    c2.metric("Features", len(FEATURES))
    c3.metric("Churn rate", f"{df['Churn'].mean():.1%}")
    st.subheader("What's inside")
    st.markdown(
        "- **Dataset Explorer**: search the data, view statistics, column profiles and missing values\n"
        "- **EDA Dashboard**: interactive charts and the correlation heatmap\n"
        "- **Model Training**: train any of six models and see its metrics\n"
        "- **Model Comparison**: leaderboard of all models, including tuned versions, and the champion\n"
        "- **Prediction System**: enter a customer's details and get a churn prediction"
    )
    st.caption("Dataset: Telco Customer Churn (Kaggle).")


def explorer():
    st.header("Dataset Explorer")
    t1, t2, t3, t4 = st.tabs(["Data", "Statistics", "Column profile", "Missing values"])
    with t1:
        q = st.text_input("Search (matches any column)")
        view = df
        if q:
            mask = df.astype(str).apply(
                lambda s: s.str.contains(q, case=False, regex=False, na=False)).any(axis=1)
            view = df[mask]
        st.caption(f"{len(view):,} of {len(df):,} rows")
        st.dataframe(view)
    with t2:
        st.dataframe(df.describe().T)
    with t3:
        profile = pd.DataFrame({"dtype": df.dtypes.astype(str),
                                "unique values": df.nunique(),
                                "missing": df.isna().sum()})
        st.dataframe(profile)
    with t4:
        st.dataframe(df.isna().sum().rename("Missing values").to_frame())
        st.caption("TotalCharges is blank for customers with tenure 0; the model pipeline imputes it.")


def eda():
    st.header("EDA Dashboard")
    plot_df = df.copy()
    plot_df["Churn"] = plot_df["Churn"].map({0: "Retained", 1: "Churned"})
    plot_df["SeniorCitizen"] = plot_df["SeniorCitizen"].map({0: "No", 1: "Yes"})

    st.subheader("Feature distribution")
    col = st.selectbox("Feature", FEATURES)
    split = st.checkbox("Split by churn", value=True)
    numeric = col in utils.NUM_COLS
    fig = px.histogram(plot_df, x=col, color="Churn" if split else None,
                       barmode="overlay" if numeric else "group",
                       nbins=40 if numeric else None,
                       opacity=0.75 if numeric else 1.0)
    st.plotly_chart(fig)

    st.subheader("Churn rate by category")
    cat = st.selectbox("Category", CAT_COLS + ["SeniorCitizen"])
    rate = df.groupby(cat)["Churn"].mean().reset_index()
    fig2 = px.bar(rate, x=cat, y="Churn", labels={"Churn": "Churn rate"})
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2)

    st.subheader("Correlation heatmap")
    corr = df[utils.NUM_COLS + ["SeniorCitizen", "Churn"]].corr()
    st.plotly_chart(px.imshow(corr, text_auto=".2f", zmin=-1, zmax=1,
                              color_continuous_scale="RdBu_r"))


def training():
    st.header("Model Training")
    st.write("Pick a model and train it on the same 80/20 stratified split used in the notebook.")
    name = st.selectbox("Model", utils.MODEL_NAMES)
    if name == "SVM":
        st.info("SVM can take 30-60 seconds to train.")
    if st.button("Train model"):
        with st.spinner(f"Training {name}..."):
            out = utils.train_model(name)
        cols = st.columns(5)
        for c, m in zip(cols, METRICS):
            c.metric(m, f"{out['metrics'][m]:.3f}")
        cm = confusion_matrix(out["y_test"], out["pred"])
        labels = ["Retained", "Churn"]
        st.plotly_chart(px.imshow(cm, text_auto=True, x=labels, y=labels,
                                  labels=dict(x="Predicted", y="Actual"),
                                  color_continuous_scale="Blues",
                                  title=f"Confusion matrix: {name}"))


def comparison():
    st.header("Model Comparison")
    results = utils.load_results()
    champ = results[results["Model"] == utils.CHAMPION].iloc[0]
    st.success(f"Champion model: **{utils.CHAMPION}** | F1 {champ['F1']:.3f} | "
               f"Recall {champ['Recall']:.3f} | ROC-AUC {champ['ROC-AUC']:.3f}")

    st.subheader("Leaderboard")
    board = results.sort_values("F1", ascending=False).reset_index(drop=True)
    st.dataframe(board.style.highlight_max(subset=METRICS, color="#c8e6c9")
                 .format({m: "{:.3f}" for m in METRICS}))

    st.subheader("Metric comparison")
    chosen = st.multiselect("Metrics", METRICS, default=["Accuracy", "F1"])
    if chosen:
        chart_df = results.copy()
        chart_df["Model"] = chart_df["Model"].apply(
            lambda m: f"★ {m}" if m == utils.CHAMPION else m)
        long = chart_df.melt(id_vars="Model", value_vars=chosen,
                             var_name="Metric", value_name="Score")
        st.plotly_chart(px.bar(long, x="Model", y="Score", color="Metric", barmode="group"))

    for title, fname in [("ROC and precision-recall curves", "roc_pr_curves.png"),
                         ("Confusion matrices", "confusion_matrices.png"),
                         ("Feature importance (Random Forest)", "feature_importance.png")]:
        path = utils.VISUALS_DIR / fname
        if path.exists():
            st.subheader(title)
            st.image(str(path))


def prediction():
    st.header("Prediction System")
    st.write("Enter a customer's details to estimate their churn risk.")
    model = utils.load_model()

    def opts(c):
        return sorted(df[c].unique().tolist())

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            gender = st.selectbox("Gender", opts("gender"))
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Partner", opts("Partner"))
            dependents = st.selectbox("Dependents", opts("Dependents"))
            tenure = st.slider("Tenure (months)", 0, 72, 12)
            phone = st.selectbox("Phone service", opts("PhoneService"))
            lines = st.selectbox("Multiple lines", opts("MultipleLines"))
        with c2:
            internet = st.selectbox("Internet service", opts("InternetService"))
            security = st.selectbox("Online security", opts("OnlineSecurity"))
            backup = st.selectbox("Online backup", opts("OnlineBackup"))
            protection = st.selectbox("Device protection", opts("DeviceProtection"))
            support = st.selectbox("Tech support", opts("TechSupport"))
            tv = st.selectbox("Streaming TV", opts("StreamingTV"))
            movies = st.selectbox("Streaming movies", opts("StreamingMovies"))
        with c3:
            contract = st.selectbox("Contract", opts("Contract"))
            paperless = st.selectbox("Paperless billing", opts("PaperlessBilling"))
            payment = st.selectbox("Payment method", opts("PaymentMethod"))
            monthly = st.slider("Monthly charges ($)", 18.0, 120.0, 70.0, 0.05)
        submitted = st.form_submit_button("Predict")

    if submitted:
        total = np.nan if tenure == 0 else tenure * monthly  # estimated
        row = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner, "Dependents": dependents, "tenure": tenure,
            "PhoneService": phone, "MultipleLines": lines,
            "InternetService": internet, "OnlineSecurity": security,
            "OnlineBackup": backup, "DeviceProtection": protection,
            "TechSupport": support, "StreamingTV": tv, "StreamingMovies": movies,
            "Contract": contract, "PaperlessBilling": paperless,
            "PaymentMethod": payment, "MonthlyCharges": monthly,
            "TotalCharges": total,
        }])[FEATURES]

        proba = float(model.predict_proba(row)[0][1])
        churn = proba >= 0.5
        confidence = proba if churn else 1 - proba
        if churn:
            st.error("**Churn Risk**")
        else:
            st.success("**Retained Account**")
        m1, m2 = st.columns(2)
        m1.metric("Confidence", f"{confidence:.1%}")
        m2.metric("Churn probability", f"{proba:.1%}")
        st.progress(proba)
        st.caption("Total charges are estimated as tenure x monthly charges. The model was trained "
                   "with class weighting, so treat the probability as a risk score, not an exact frequency.")


PAGES = {
    "Home": home,
    "Dataset Explorer": explorer,
    "EDA Dashboard": eda,
    "Model Training": training,
    "Model Comparison": comparison,
    "Prediction System": prediction,
}
choice = st.sidebar.radio("Navigate", list(PAGES))
PAGES[choice]()
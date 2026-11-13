import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st

st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
M = Path(__file__).parent / "models"


@st.cache_resource
def load():
    pre = joblib.load(M / "preprocessor.joblib")
    schema = json.load(open(M / "schema.json"))
    metrics = json.load(open(M / "metrics.json"))
    models = {}
    for n in metrics["models"]:
        if n == "1D CNN":
            from tensorflow import keras
            models[n] = keras.models.load_model(M / "cnn.keras")
        else:
            models[n] = joblib.load(M / f"{n.lower().replace(' ', '_')}.joblib")
    return pre, schema, metrics, models, np.load(M / "sample_X.npy")


pre, schema, metrics, models, sample = load()
names = schema["features"]


def clean(df):
    df = df[schema["columns"]].copy()
    for c in schema["numeric"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    for c in schema["categorical"]:
        df[c] = df[c].fillna("Unknown").astype(str)
    return df


def proba(name, Xt):
    m = models[name]
    if name == "1D CNN":
        return m.predict(Xt.astype("float32")[..., None], verbose=0).ravel()
    return m.predict_proba(Xt)[:, 1]


@st.cache_resource
def explainer():
    return shap.TreeExplainer(models["Random Forest"])


def shap_values(Xt):
    v = explainer().shap_values(Xt, check_additivity=False)
    if isinstance(v, list):
        v = v[1]
    v = np.asarray(v)
    return v[:, :, 1] if v.ndim == 3 else v


st.title("📉 Customer Churn Prediction")
model_name = st.sidebar.selectbox("Model", list(models), index=list(models).index(
    max(metrics["models"], key=lambda n: metrics["models"][n]["roc_auc"])))
threshold = st.sidebar.slider("Churn threshold", 0.1, 0.9, 0.5, 0.05)
st.sidebar.caption("Lower threshold = catch more churners, more false alarms.")

tab1, tab2, tab3, tab4 = st.tabs(
    ["Single prediction", "Bulk CSV prediction", "Model comparison", "Explainability (SHAP)"])

# ---------- Single prediction ----------
with tab1:
    cols = st.columns(3)
    inputs = {}
    for i, (c, (lo, hi, med)) in enumerate(schema["numeric"].items()):
        kw = {} if lo == hi else {"min_value": lo, "max_value": hi}
        inputs[c] = cols[i % 3].number_input(c, value=float(med), **kw)
    for i, (c, opts) in enumerate(schema["categorical"].items(), start=len(schema["numeric"])):
        inputs[c] = cols[i % 3].selectbox(c, opts)

    if st.button("Predict churn", type="primary"):
        Xt = pre.transform(clean(pd.DataFrame([inputs])))
        p = float(proba(model_name, Xt)[0])
        (st.error if p >= threshold else st.success)(
            f"{'Likely to CHURN' if p >= threshold else 'Likely to STAY'}  |  churn probability {p:.1%}")
        st.progress(min(p, 1.0))

        sv = shap_values(Xt)[0]
        s = pd.Series(sv, index=names)
        top = s.reindex(s.abs().sort_values(ascending=False).head(10).index)[::-1]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(top.index, top.values, color=["#d62728" if x > 0 else "#1f77b4" for x in top.values])
        ax.set_xlabel("SHAP value (red = pushes toward churn)")
        ax.set_title("Why this prediction? (Random Forest explanation)")
        st.pyplot(fig)

# ---------- Bulk CSV ----------
with tab2:
    st.write("Upload a CSV with the same columns as the training data (the Churn column is optional).")
    st.download_button("Download CSV template", pd.DataFrame(columns=schema["columns"]).to_csv(index=False),
                       "template.csv")
    up = st.file_uploader("Customer CSV", type="csv")
    if up:
        raw = pd.read_csv(up)
        missing = [c for c in schema["columns"] if c not in raw.columns]
        if missing:
            st.error(f"Missing columns: {missing}")
        else:
            p = proba(model_name, pre.transform(clean(raw)))
            out = raw.copy()
            out["Churn_Probability"] = p.round(3)
            out["Churn_Prediction"] = np.where(p >= threshold, "Yes", "No")
            out["Risk"] = pd.cut(p, [0, 0.33, 0.66, 1.0], labels=["Low", "Medium", "High"], include_lowest=True)
            c1, c2 = st.columns(2)
            c1.metric("Customers", len(out))
            c2.metric("Predicted to churn", int((p >= threshold).sum()))
            st.dataframe(out)
            st.download_button("Download predictions", out.to_csv(index=False), "churn_predictions.csv")

# ---------- Model comparison ----------
with tab3:
    info = metrics["info"]
    st.caption(f"{info['rows']} customers | {info['churn_rate']:.1%} churn | "
               f"{info['n_train']} train / {info['n_test']} test (stratified split)")
    table = pd.DataFrame({n: {k: v for k, v in m.items() if not isinstance(v, (list, dict))}
                          for n, m in metrics["models"].items()}).T
    st.dataframe(table.round(3))
    st.success(f"Best ROC-AUC: {table['roc_auc'].idxmax()} ({table['roc_auc'].max():.3f})")
    metric = st.selectbox("Compare models by", ["roc_auc", "accuracy", "precision", "recall", "f1"])
    st.bar_chart(table[metric])

    c1, c2 = st.columns(2)
    with c1:
        fig, ax = plt.subplots(figsize=(4, 3))
        for n, m in metrics["models"].items():
            ax.plot(m["roc"]["fpr"], m["roc"]["tpr"], label=f"{n} ({m['roc_auc']:.2f})")
        ax.plot([0, 1], [0, 1], "k--")
        ax.set_xlabel("False positive rate")
        ax.set_ylabel("True positive rate")
        ax.legend(fontsize=6)
        st.pyplot(fig, use_container_width=False)

    with c2:
        cm = np.array(metrics["models"][model_name]["confusion_matrix"])
        fig, ax = plt.subplots(figsize=(3, 2.6))
        ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, v, ha="center", va="center")
        ax.set_xticks([0, 1], ["Stay", "Churn"])
        ax.set_yticks([0, 1], ["Stay", "Churn"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title(f"Confusion matrix: {model_name}", fontsize=9)
        st.pyplot(fig, use_container_width=False)

# ---------- SHAP ----------
with tab4:
    st.write("SHAP shows how much each feature pushes a prediction toward churn (Random Forest).")
    sv = shap_values(sample[:200])
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Average importance")
        plt.figure()
        shap.summary_plot(sv, sample[:200], feature_names=names, plot_type="bar", show=False)
        plt.xlabel("Average impact on churn prediction")
        plt.tight_layout()
        st.pyplot(plt.gcf())
        plt.close()
    with c2:
        st.subheader("Impact of each feature")
        plt.figure()
        shap.summary_plot(sv, sample[:200], feature_names=names, show=False)
        st.pyplot(plt.gcf())
        plt.close()
import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from lime.lime_tabular import LimeTabularExplainer

st.set_page_config(page_title="SIC Platelet 4-Class Group Prediction (XGBoost)", layout="wide")

APP_DIR = Path(__file__).resolve().parent
MODEL_PATH = APP_DIR / "XGB.pkl"   # exported by the training script
XTEST_PATH = APP_DIR / "X_test.csv"

@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_x_test():
    return pd.read_csv(XTEST_PATH)

st.title("SIC Platelet 4-Class Group Prediction (XGBoost)")
st.caption(
    "Cloud app is for inference only: run train_export_xgb_4class.py locally to generate "
    "XGB.pkl and X_test.csv, then push them to GitHub."
)

# ---- Safe loading ----
try:
    bundle = load_bundle()
except Exception as e:
    st.error(
        "Failed to load XGB.pkl. Please ensure it is committed to the repo root and that "
        "requirements.txt includes dependencies such as xgboost / imbalanced-learn."
    )
    st.exception(e)
    st.stop()

try:
    X_test = load_x_test()
except Exception as e:
    st.warning(
        "Failed to load X_test.csv (LIME explanation will be unavailable). "
        "Please ensure X_test.csv is in the repo root."
    )
    X_test = None

model = bundle["model"]
feature_cols = list(bundle["feature_cols"])
classes = [str(c) for c in bundle["classes"]]
n_classes = int(bundle.get("n_classes", len(classes)))
train_median = pd.Series(bundle.get("train_median", pd.Series(dtype=float)))

def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """Basic preprocessing aligned with training: numeric coercion, missing fill, type casting."""
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.apply(pd.to_numeric, errors="coerce")
    if not train_median.empty:
        df = df.fillna(train_median)
    df = df.fillna(0).astype(np.float32)
    return df

# -------- Input UI (24 features) --------
st.subheader("Input features (24)")
with st.form("input_form"):
    cols = st.columns(3)
    inputs = {}
    for i, feat in enumerate(feature_cols):
        with cols[i % 3]:
            dv = float(train_median.get(feat, 0.0)) if not train_median.empty else 0.0
            inputs[feat] = st.number_input(feat, value=dv, step=0.1, format="%.4f")
    submitted = st.form_submit_button("Predict")

if submitted:
    x_row = pd.DataFrame([inputs], columns=feature_cols)
    x_row = preprocess(x_row)

    # Prediction
    try:
        proba = model.predict_proba(x_row)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = classes[pred_idx] if pred_idx < len(classes) else str(pred_idx)
    except Exception as e:
        st.error(
            "Prediction failed. Please verify that the model and feature column order match "
            "(24 columns), and that the model file is valid."
        )
        st.exception(e)
        st.stop()

    st.subheader("Prediction")
    c1, c2 = st.columns([1, 1])
    c1.metric("Predicted class", pred_label)
    c2.metric("Top probability", f"{float(proba[pred_idx]):.4f}")

    df_prob = (
        pd.DataFrame({"Class": classes, "Probability": [float(p) for p in proba]})
        .sort_values("Probability", ascending=False)
        .reset_index(drop=True)
    )
    st.dataframe(df_prob, use_container_width=True)

    # -------- SHAP (single instance) --------
    st.subheader("SHAP explanation (single instance, for the predicted class)")
    try:
        import shap  # deferred import

        # If model is a pipeline, try to extract xgb step; otherwise use model directly
        if hasattr(model, "named_steps") and "xgb" in model.named_steps:
            xgb = model.named_steps["xgb"]
        else:
            xgb = model

        explainer = shap.TreeExplainer(xgb)
        shap_exp = explainer(x_row)  # shap.Explanation

        vals = shap_exp.values if hasattr(shap_exp, "values") else shap_exp
        vals = np.asarray(vals)

        # Typical multi-class shapes: (1, n_features, n_classes)
        if vals.ndim == 3:
            contrib = vals[0, :, pred_idx]
        elif vals.ndim == 2:
            contrib = vals[0, :]
        else:
            contrib = vals.reshape(-1)

        feat_names = np.array(feature_cols)
        top_n = min(15, len(contrib))
        top_idx = np.argsort(np.abs(contrib))[-top_n:]

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh(feat_names[top_idx], contrib[top_idx])
        ax.set_title(f"Top {top_n} SHAP contributions | Predicted={pred_label}")
        ax.set_xlabel("SHAP value (impact on model output)")
        plt.tight_layout()
        st.pyplot(fig)
    except Exception as e:
        st.warning("SHAP generation failed (often due to environment/version mismatch).")
        st.exception(e)

    # -------- LIME (multi-class) --------
    st.subheader("LIME explanation (single instance)")
    if X_test is None:
        st.info("X_test.csv is not available, so LIME is disabled.")
    else:
        try:
            # Background data must align with feature_cols
            X_bg = X_test[feature_cols].copy()
            X_bg = preprocess(X_bg)

            explainer = LimeTabularExplainer(
                training_data=X_bg.values,
                feature_names=feature_cols,
                class_names=classes,
                mode="classification",
            )

            def predict_fn(x_np):
                x_df = pd.DataFrame(x_np, columns=feature_cols)
                x_df = preprocess(x_df)
                return model.predict_proba(x_df)

            exp = explainer.explain_instance(
                data_row=x_row.values[0],
                predict_fn=predict_fn,
                num_features=min(15, len(feature_cols)),
            )
            st.components.v1.html(exp.as_html(show_table=False), height=800, scrolling=True)
        except Exception as e:
            st.warning("LIME generation failed.")
            st.exception(e)

# ---- Optional: show training-time figures (if you pushed the results folder to GitHub) ----
with st.expander("View training-time evaluation figures / merged SHAP plots (if results folder exists)", expanded=False):
    results_dir = APP_DIR / "results_dev_4class_shap_bee_left_bar_right"
    if results_dir.exists():
        imgs = sorted(list(results_dir.glob("*.png")))
        if imgs:
            for p in imgs:
                st.image(str(p), caption=p.name, use_container_width=True)
        else:
            st.info("Results folder detected, but no PNG files found.")
    else:
        st.info("Folder results_dev_4class_shap_bee_left_bar_right not found.")

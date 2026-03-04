# -*- coding: utf-8 -*-
# =============================================================================
#  Heart Disease Predictive Analytics - Streamlit Dashboard
#  Run with: streamlit run app.py
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import io
import os

# -- Page Config --------------------------------------------------------------
st.set_page_config(
    page_title="Heart Disease Risk Predictor",
    page_icon="heart",
    layout="wide",
    initial_sidebar_state="expanded"
)


# -- Load Resources -----------------------------------------------------------
@st.cache_resource
def load_resources():
    model       = joblib.load("heart_disease_model.pkl")
    scaler      = joblib.load("heart_disease_scaler.pkl")
    feat_imp_df = joblib.load("feature_importance.pkl")
    with open("model_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    return model, scaler, feat_imp_df, metadata


@st.cache_data
def load_dataset():
    if os.path.exists("heart.csv"):
        return pd.read_csv("heart.csv")
    return None


model, scaler, feat_imp_df, metadata = load_resources()
df_raw = load_dataset()


# -- Header -------------------------------------------------------------------
st.title("AI-Powered Heart Disease Risk Predictor")
st.markdown("""
This dashboard uses a machine learning model trained on the **Heart Disease UCI dataset**
to predict a patient's risk of heart disease based on clinical parameters.
Adjust the parameters in the sidebar and click **Predict** to get a risk assessment.
""")


# -- Sidebar: Patient Inputs --------------------------------------------------
st.sidebar.header("Patient Parameters")
st.sidebar.markdown("Adjust the sliders to match patient values:")

age      = st.sidebar.slider("Age", 29, 80, 50, help="Patient age in years")
sex      = st.sidebar.selectbox("Sex", ["Male (1)", "Female (0)"])
cp       = st.sidebar.selectbox("Chest Pain Type", [
               "Typical Angina (0)", "Atypical Angina (1)",
               "Non-Anginal Pain (2)", "Asymptomatic (3)"])
trestbps = st.sidebar.slider("Resting BP (mmHg)", 80, 200, 120)
chol     = st.sidebar.slider("Cholesterol (mg/dl)", 100, 600, 240)
fbs      = st.sidebar.selectbox("Fasting Blood Sugar > 120?", ["No (0)", "Yes (1)"])
restecg  = st.sidebar.selectbox("Resting ECG", [
               "Normal (0)", "ST Abnormality (1)", "LV Hypertrophy (2)"])
thalach  = st.sidebar.slider("Max Heart Rate", 60, 210, 150)
exang    = st.sidebar.selectbox("Exercise Angina?", ["No (0)", "Yes (1)"])
oldpeak  = st.sidebar.slider("ST Depression", 0.0, 7.0, 1.0, 0.1)
slope    = st.sidebar.selectbox("ST Slope", [
               "Upsloping (0)", "Flat (1)", "Downsloping (2)"])
ca       = st.sidebar.slider("Major Vessels (0-3)", 0, 3, 0)
thal     = st.sidebar.selectbox("Thalassemia", [
               "Normal (1)", "Fixed Defect (2)", "Reversible Defect (3)"])


# -- Helper: Parse Sidebar Inputs ---------------------------------------------
def parse_input():
    return {
        'age'     : age,
        'sex'     : int(sex.split("(")[1][0]),
        'cp'      : int(cp.split("(")[1][0]),
        'trestbps': trestbps,
        'chol'    : chol,
        'fbs'     : int(fbs.split("(")[1][0]),
        'restecg' : int(restecg.split("(")[1][0]),
        'thalach' : thalach,
        'exang'   : int(exang.split("(")[1][0]),
        'oldpeak' : oldpeak,
        'slope'   : int(slope.split("(")[1][0]),
        'ca'      : ca,
        'thal'    : int(thal.split("(")[1][0]),
    }


def preprocess_input(raw: dict) -> pd.DataFrame:
    """Replicate the training preprocessing pipeline for a single patient input."""
    df_in = pd.DataFrame([raw])

    # One-hot encode multi-class categoricals (matching training)
    for col in ['cp', 'restecg', 'slope', 'thal']:
        if col in df_in.columns:
            dummies = pd.get_dummies(df_in[col], prefix=col)
            df_in = pd.concat([df_in.drop(col, axis=1), dummies], axis=1)

    # Align with training features
    train_features = metadata['features']
    for col in train_features:
        if col not in df_in.columns:
            df_in[col] = 0
    df_in = df_in[train_features]

    # Scale continuous features
    cont = metadata['continuous_cols']
    df_in[cont] = scaler.transform(df_in[cont])
    return df_in


# -- Main: Tabs ---------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["Risk Prediction", "Feature Importance", "Dataset Summary"])


# -- Tab 1: Prediction --------------------------------------------------------
with tab1:
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("Patient Summary")
        raw_input  = parse_input()
        summary_df = pd.DataFrame(raw_input.items(), columns=['Parameter', 'Value'])
        st.dataframe(summary_df, use_container_width=True, hide_index=True)

    with col2:
        st.subheader("Risk Assessment")
        if st.button("Predict Heart Disease Risk", use_container_width=True):
            try:
                X_in       = preprocess_input(raw_input)
                pred_label = model.predict(X_in)[0]
                pred_prob  = model.predict_proba(X_in)[0][1]

                # Risk display
                if pred_prob < 0.3:
                    risk_label, risk_color = "Low Risk",      "green"
                elif pred_prob < 0.6:
                    risk_label, risk_color = "Moderate Risk", "orange"
                else:
                    risk_label, risk_color = "High Risk",     "red"

                st.markdown(f"### {risk_label}")
                st.metric("Predicted Probability", f"{pred_prob:.1%}")
                st.metric("Model Prediction",
                          "Heart Disease Detected" if pred_label == 1 else "No Heart Disease")
                st.progress(float(pred_prob))

                # Actionable recommendations
                st.markdown("---")
                st.markdown("#### Recommendations")
                if pred_prob >= 0.6:
                    st.error("High risk detected. Immediate cardiology referral recommended.")
                    st.markdown(
                        "- Schedule stress ECG and echocardiogram\n"
                        "- Review lipid-lowering and antihypertensive medications\n"
                        "- Urgent dietary and lifestyle intervention"
                    )
                elif pred_prob >= 0.3:
                    st.warning("Moderate risk. Preventive measures advised.")
                    st.markdown(
                        "- Annual cardiovascular check-ups\n"
                        "- Increase physical activity (150 min/week)\n"
                        "- Heart-healthy diet (Mediterranean-style)"
                    )
                else:
                    st.success("Low risk. Maintain healthy lifestyle.")
                    st.markdown(
                        "- Continue regular exercise\n"
                        "- Maintain healthy weight and diet\n"
                        "- Routine check-up every 2 years"
                    )

                # Save result for download
                result_df = pd.DataFrame([{
                    **raw_input,
                    'predicted_label' : pred_label,
                    'risk_probability': round(pred_prob, 4),
                    'risk_category'   : risk_label
                }])
                st.session_state['result_df'] = result_df

            except Exception as e:
                st.error(f"Prediction error: {e}")

        if 'result_df' in st.session_state:
            st.markdown("---")
            csv_buffer = io.StringIO()
            st.session_state['result_df'].to_csv(csv_buffer, index=False)
            st.download_button(
                label="Download Prediction as CSV",
                data=csv_buffer.getvalue(),
                file_name="heart_disease_prediction.csv",
                mime="text/csv",
                use_container_width=True
            )

    # Model performance banner
    st.markdown("---")
    m1, m2, m3 = st.columns(3)
    m1.metric("Model Accuracy", f"{metadata['metrics']['accuracy']:.1%}")
    m2.metric("ROC-AUC Score",  f"{metadata['metrics']['roc_auc']:.4f}")
    m3.metric("F1-Score",       f"{metadata['metrics']['f1']:.4f}")
    st.caption(f"Model: {metadata['best_model']} | Dataset: Heart Disease UCI")


# -- Tab 2: Feature Importance ------------------------------------------------
with tab2:
    st.subheader("Top Predictive Risk Factors")
    st.markdown("Features ranked by their contribution to the model's predictions:")

    top_n     = st.slider("Show top N features:", 5, min(20, len(feat_imp_df)), 10)
    top_feats = feat_imp_df.head(top_n).sort_values('importance')

    fig, ax = plt.subplots(figsize=(9, max(4, top_n * 0.4)))
    palette = sns.color_palette('YlOrRd_r', top_n)
    bars    = ax.barh(top_feats['feature'], top_feats['importance'], color=palette)
    ax.set_xlabel('Importance Score')
    ax.set_title(f'Top {top_n} Risk Factors - {metadata["best_model"]}')
    for bar, val in zip(bars, top_feats['importance']):
        ax.text(bar.get_width() + 0.0005, bar.get_y() + bar.get_height() / 2,
                f'{val:.4f}', va='center', fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)

    st.dataframe(
        feat_imp_df.rename(columns={'importance': 'Score', 'importance_pct': '%'}).head(top_n),
        use_container_width=True,
        hide_index=True
    )


# -- Tab 3: Dataset Summary ---------------------------------------------------
with tab3:
    st.subheader("Dataset Overview")
    if df_raw is not None:
        c1, c2 = st.columns(2)
        c1.metric("Total Records", df_raw.shape[0])
        c2.metric("Features",      df_raw.shape[1] - 1)

        st.markdown("**Class Distribution**")
        target_col = 'target' if 'target' in df_raw.columns else df_raw.columns[-1]
        vc = df_raw[target_col].value_counts().rename({0: 'No Disease', 1: 'Heart Disease'})
        st.bar_chart(vc)

        st.markdown("**Summary Statistics**")
        st.dataframe(df_raw.describe().round(2), use_container_width=True)

        st.markdown("**Raw Data Sample**")
        st.dataframe(df_raw.head(20), use_container_width=True)

        csv_buf = io.StringIO()
        df_raw.to_csv(csv_buf, index=False)
        st.download_button(
            "Download Full Dataset",
            csv_buf.getvalue(),
            "heart_disease_data.csv",
            "text/csv"
        )
    else:
        st.info("Dataset file 'heart.csv' not found. Run the notebook first to generate the model.")


# -- Footer -------------------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray; font-size: 0.85em;'>"
    "Heart Disease Risk Predictor | Built with scikit-learn and Streamlit | "
    "<b>For educational purposes only - not a substitute for medical advice.</b>"
    "</div>",
    unsafe_allow_html=True
)
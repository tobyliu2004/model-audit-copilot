# dashboard/fairness_app.py

import streamlit as st
import pandas as pd
from copilot.fairness.fairness_audit import audit_group_fairness

st.title("Fairness Auditing")
st.write("Upload a prediction dataset with actual, predicted, and sensitive group columns.")

uploaded_file = st.file_uploader("Upload predictions CSV", type="csv")

if uploaded_file:
    try:
        df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
        st.write("Preview of your data:")
        st.dataframe(df.head())

        # Detect valid columns
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        all_cols = df.columns.tolist()

        # Select column mappings
        y_true_col = st.selectbox("Select ground truth column (y_true)", numeric_cols)
        y_pred_col = st.selectbox("Select prediction column (y_pred)", numeric_cols)
        group_col = st.selectbox("Select sensitive attribute (e.g., race, gender)", all_cols)

        if st.button("Run Fairness Audit"):
            report = audit_group_fairness(
                y_true=df[y_true_col],
                y_pred=df[y_pred_col],
                sensitive_feature=df[group_col]
            )
            st.subheader("Fairness Report")
            st.dataframe(report)

            csv = report.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="Download Report as CSV",
                data=csv,
                file_name="fairness_report.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Failed to process file: {e}")

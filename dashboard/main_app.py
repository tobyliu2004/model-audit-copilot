# dashboard/main_app.py

import streamlit as st
import pandas as pd
from copilot.drift.drift_detector import compare_datasets
from copilot.fairness.fairness_audit import audit_group_fairness
from copilot.auditor.audit_orchestrator import AuditOrchestrator
import joblib

st.set_page_config(page_title="Model Audit Copilot", layout="wide")
st.title("Model Audit Copilot")

tabs = st.tabs([
    "📊 Drift Detection", 
    "⚖️ Fairness Audit", 
    "🧭 Outlier Detection", 
    "🛑 Leakage Detection", 
    "📐 Schema Consistency",
    "🧠 Explainability",
    "📊 Model Comparison"
])


# === Tab 1: Drift Detection ===
with tabs[0]:
    st.header("Drift Detection")
    ref_file = st.file_uploader("Upload reference dataset", type="csv", key="ref")
    cur_file = st.file_uploader("Upload current dataset", type="csv", key="cur")

    if ref_file and cur_file:
        try:
            ref_df = pd.read_csv(ref_file)
            cur_df = pd.read_csv(cur_file)

            method = st.selectbox("Drift Detection Method", ["ks", "psi"])

            auditor = AuditOrchestrator(ref_df, cur_df)
            auditor.run_drift_check(method=method)
            summary_df = auditor.get_results()["drift"]

            st.subheader("Drift Summary")
            st.dataframe(summary_df)

            drifted = summary_df['drift_detected'].sum()
            total = summary_df.shape[0]
            st.info(f"Drift detected in {drifted} out of {total} features.")

            csv = summary_df.to_csv(index=True).encode('utf-8')
            st.download_button("Download Report", data=csv, file_name="drift_report.csv")

            fig = auditor.get_drift_plot()
            if fig:
                st.subheader("Drift Visualization")
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error processing files: {e}")

# === Tab 2: Fairness Auditing ===
with tabs[1]:
    st.header("Fairness Audit")
    pred_file = st.file_uploader("Upload prediction dataset", type="csv", key="fair")

    if pred_file:
        try:
            df = pd.read_csv(pred_file)
            st.write("Preview:")
            st.dataframe(df.head())

            numeric_cols = df.select_dtypes(include='number').columns.tolist()
            all_cols = df.columns.tolist()

            y_true_col = st.selectbox("Select true target column", numeric_cols)
            y_pred_col = st.selectbox("Select predicted column", numeric_cols)
            group_col = st.selectbox("Select sensitive group column", all_cols)

            if st.button("Run Fairness Audit"):
                report = audit_group_fairness(
                    y_true=df[y_true_col],
                    y_pred=df[y_pred_col],
                    sensitive_feature=df[group_col]
                )
                st.subheader("Fairness Audit Report")
                st.dataframe(report)

                csv = report.to_csv(index=False).encode('utf-8')
                st.download_button("Download Fairness Report", data=csv, file_name="fairness_report.csv")

        except Exception as e:
            st.error(f"Error processing predictions file: {e}")

# === Tab 3: Outlier Detection ===
with tabs[2]:
    st.header("Outlier Detection")
    outlier_file = st.file_uploader("Upload dataset for anomaly detection", type="csv", key="outlier")

    if outlier_file:
        try:
            df = pd.read_csv(outlier_file)
            st.write("Preview:")
            st.dataframe(df.head())

            num_outliers = st.slider("Select number of top outliers to view", min_value=5, max_value=50, value=10)

            from copilot.outliers.outlier_detector import detect_outliers
            outliers = detect_outliers(df, n_outliers=num_outliers)

            st.subheader("Top Outliers")
            st.dataframe(outliers)

            csv = outliers.to_csv(index=False).encode("utf-8")
            st.download_button("Download Outlier Report", data=csv, file_name="outliers.csv")

        except Exception as e:
            st.error(f"Failed to process file: {e}")

# === Tab 4: Leakage Detection ===
with tabs[3]:
    st.header("Data Leakage Detection")
    leak_file = st.file_uploader("Upload dataset to check for leakage", type="csv", key="leak")
    ref_file = st.file_uploader("Optional: Upload reference (train) dataset for overlap check", type="csv", key="leak_ref")

    if leak_file:
        try:
            cur_df = pd.read_csv(leak_file)
            ref_df = pd.read_csv(ref_file) if ref_file else None

            st.write("Preview of dataset:")
            st.dataframe(cur_df.head())

            target_col = st.selectbox("Select the target column", cur_df.columns)

            from copilot.auditor.audit_orchestrator import AuditOrchestrator
            auditor = AuditOrchestrator(ref_df, cur_df)
            auditor.run_leakage_check(dataset=cur_df, target_col=target_col)
            leakage = auditor.get_leakage_report()

            st.subheader("Target Leakage")
            st.dataframe(leakage["target_leakage"])

            st.subheader("Duplicate Rows")
            st.info(f"{leakage['duplicate_rows']} duplicate rows found.")

            st.subheader("ID-like Columns")
            st.write(leakage["id_like_columns"])

            if ref_file:
                st.subheader("Train/Test Overlap")
                st.json(leakage["train_test_overlap"])

        except Exception as e:
            st.error(f"Error processing file: {e}")

# === Tab 5: Schema Consistency ===
with tabs[4]:
    st.header("Schema & Type Consistency Check")
    ref_file = st.file_uploader("Upload reference dataset (e.g., training data)", type="csv", key="schema_ref")
    cur_file = st.file_uploader("Upload current dataset (e.g., production data)", type="csv", key="schema_cur")

    if ref_file and cur_file:
        try:
            ref_df = pd.read_csv(ref_file)
            cur_df = pd.read_csv(cur_file)

            from copilot.auditor.audit_orchestrator import AuditOrchestrator
            auditor = AuditOrchestrator(ref_df, cur_df)
            auditor.run_schema_check()
            schema = auditor.get_schema_report()

            st.subheader("Column Type Comparison")
            st.dataframe(schema["schema_comparison"])

            st.subheader("Missing / Extra Columns")
            st.json(schema["column_diff"])

            csv = schema["schema_comparison"].to_csv(index=False).encode("utf-8")
            st.download_button("Download Schema Comparison CSV", data=csv, file_name="schema_comparison.csv")

        except Exception as e:
            st.error(f"Error processing files: {e}")

# === Tab 6: Explainability ===
with tabs[5]:
    st.header("SHAP Explainability")
    st.write("Upload a trained model (.pkl) and its corresponding input dataset (.csv).")

    model_file = st.file_uploader("Upload trained model (pickle)", type="pkl")
    data_file = st.file_uploader("Upload feature dataset (CSV)", type="csv", key="shap_data")

    if model_file and data_file:
        try:
            import pickle
            import pandas as pd
            import shap

            # Load model and data
            model = pickle.load(model_file)
            df = pd.read_csv(data_file)

            # Drop non-feature columns if needed (user-controlled)
            drop_cols = st.multiselect("Columns to exclude from SHAP", df.columns)
            X = pd.get_dummies(df.drop(columns=drop_cols)).astype(float)

            # Run explainability
            from copilot.auditor.audit_orchestrator import AuditOrchestrator
            auditor = AuditOrchestrator(None, None)
            auditor.run_explainability(model, X)

            st.subheader("SHAP Summary Plot")
            fig = auditor.get_shap_plot()
            if fig:
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Could not generate SHAP plot: {e}")

# === Tab 7: Model Comparison ===
with tabs[6]:
    st.header("Model SHAP Comparison (CSV-based)")
    st.write("Upload two SHAP summary CSVs (from TreeExplainer).")

    file_a = st.file_uploader("Upload SHAP CSV for Model A", type="csv", key="shap_a")
    file_b = st.file_uploader("Upload SHAP CSV for Model B", type="csv", key="shap_b")

    if file_a and file_b:
        try:
            import pandas as pd
            import matplotlib.pyplot as plt

            shap_a = pd.read_csv(file_a).set_index("feature")
            shap_b = pd.read_csv(file_b).set_index("feature")

            # Join and compute delta
            comp_df = shap_a.join(shap_b, lsuffix="_a", rsuffix="_b").fillna(0)
            comp_df["delta"] = comp_df["mean_abs_shap_b"] - comp_df["mean_abs_shap_a"]
            comp_df = comp_df.sort_values("delta", ascending=False).reset_index()

            st.subheader("Top SHAP Δ Features (Model B - Model A)")
            st.dataframe(comp_df)

            # Bar plot
            st.subheader("SHAP Importance Delta Plot")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.barh(comp_df["feature"], comp_df["delta"], color="steelblue")
            ax.axvline(0, color="gray", linestyle="--")
            ax.set_xlabel("SHAP Δ (Model B - Model A)")
            ax.set_title("SHAP Change by Feature")
            st.pyplot(fig)

        except Exception as e:
            st.error(f"Failed to compare SHAP CSVs: {e}")

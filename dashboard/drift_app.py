# dashboard/drift_app.py
#To call in terminal use:
# PYTHONPATH=. streamlit run dashboard/drift_app.py

import streamlit as st
import pandas as pd
from copilot.drift.drift_detector import compare_datasets
from copilot.auditor.audit_orchestrator import AuditOrchestrator


st.title("Drift Detection Dashboard")
st.write("Upload a reference dataset and a current dataset to detect distributional drift.")

ref_file = st.file_uploader("Upload reference dataset (CSV)", type="csv")
cur_file = st.file_uploader("Upload current dataset (CSV)", type="csv")

if ref_file and cur_file:
    try:
        ref_df = pd.read_csv(ref_file)
        cur_df = pd.read_csv(cur_file)
    except Exception as e:
        st.error(f"Error reading uploaded files: {e}")
    else:
        st.success("Files uploaded successfully.")
    
        # ✅ Add method selector here
        method = st.selectbox("Drift Detection Method", ["ks", "psi"])

        # Use orchestrator to run audit
        auditor = AuditOrchestrator(ref_df, cur_df)
        auditor.run_drift_check(method=method)
        summary_df = auditor.get_results()["drift"]

        st.subheader("Drift Summary")
        st.dataframe(summary_df)

        # ✅ Drift count summary
        drifted = summary_df['drift_detected'].sum()
        total = summary_df.shape[0]
        st.info(f"Drift detected in {drifted} out of {total} numeric features.")

        csv = summary_df.to_csv(index=True).encode('utf-8')
        st.download_button(
            label="Download Drift Report as CSV",
            data=csv,
            file_name='drift_report.csv',
            mime='text/csv'
        )
    
        st.subheader("KS Statistic Plot")
        fig = auditor.get_drift_plot()
        if fig:
            st.pyplot(fig)
        st.pyplot(fig)
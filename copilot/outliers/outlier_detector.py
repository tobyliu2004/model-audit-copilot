# copilot/outliers/outlier_detector.py

import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_outliers(df: pd.DataFrame, n_outliers=10, contamination=0.01):
    numeric_df = df.select_dtypes(include='number').dropna()
    
    model = IsolationForest(n_estimators=100, contamination=contamination, random_state=42)
    model.fit(numeric_df)

    df_scores = numeric_df.copy()
    df_scores['anomaly_score'] = model.decision_function(numeric_df)
    df_scores['outlier_flag'] = model.predict(numeric_df)

    # Predict returns -1 for outliers, 1 for inliers
    outliers = df_scores[df_scores['outlier_flag'] == -1]
    top_outliers = outliers.sort_values(by='anomaly_score').head(n_outliers)

    return top_outliers.reset_index()

#example use
#from copilot.outliers.outlier_detector import detect_outliers

#top_outliers = detect_outliers(df=current_df, n_outliers=15)
#print(top_outliers)

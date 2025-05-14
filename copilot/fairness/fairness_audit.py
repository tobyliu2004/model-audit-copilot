# copilot/fairness/fairness_audit.py

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

def audit_group_fairness(y_true, y_pred, sensitive_feature: pd.Series):
    df = pd.DataFrame({
        'y_true': y_true,
        'y_pred': y_pred,
        'group': sensitive_feature
    })

    group_metrics = []

    for group in df['group'].unique():
        subset = df[df['group'] == group]
        mae = mean_absolute_error(subset['y_true'], subset['y_pred'])
        mse = mean_squared_error(subset['y_true'], subset['y_pred'])
        rmse = np.sqrt(mse)
        bias = np.mean(subset['y_pred'] - subset['y_true'])

        group_metrics.append({
            'group': group,
            'count': len(subset),
            'mae': round(mae, 4),
            'rmse': round(rmse, 4),
            'bias': round(bias, 4)
        })

    return pd.DataFrame(group_metrics)

#example use case
#from copilot.fairness.fairness_audit import audit_group_fairness

# After your model has made predictions:
#df['y_pred'] = model.predict(df[features])

#report = audit_group_fairness(
#    y_true=df['target'],
#    y_pred=df['y_pred'],
#    sensitive_feature=df['gender']  # or 'race', etc.
#)
#print(report)

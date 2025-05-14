# copilot/drift/drift_detector.py

import pandas as pd
import numpy as np
from scipy.stats import ks_2samp
import matplotlib.pyplot as plt

class DriftReport:
    def __init__(self, results: dict):
        self.results = results

    def summary(self):
        return pd.DataFrame(self.results).T

    def plot(self, return_fig=False):
        import matplotlib.pyplot as plt

        features = list(self.results.keys())
        ks_values = [self.results[f]['ks_stat'] for f in features]
        colors = ['crimson' if self.results[f]['drift_detected'] else 'steelblue' for f in features]

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(features, ks_values, color=colors)
        ax.set_xlabel("KS Statistic")
        ax.set_title("Feature-wise Drift (KS Test)")
        plt.tight_layout()

        if return_fig:
            return fig
        else:
            plt.show()

def calculate_psi(expected, actual, buckets=10):
    def _get_bins(array, buckets):
        if np.issubdtype(array.dtype, np.number):
            return np.histogram_bin_edges(array, bins=buckets)
        else:
            return sorted(set(array))

    expected = pd.Series(expected).dropna()
    actual = pd.Series(actual).dropna()

    bins = _get_bins(expected, buckets)

    if np.issubdtype(expected.dtype, np.number):
        expected_counts, _ = np.histogram(expected, bins=bins)
        actual_counts, _ = np.histogram(actual, bins=bins)
    else:
        expected_counts = expected.value_counts().reindex(bins, fill_value=0)
        actual_counts = actual.value_counts().reindex(bins, fill_value=0)

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    psi_values = (expected_pct - actual_pct) * np.log((expected_pct + 1e-6) / (actual_pct + 1e-6))
    return round(np.sum(psi_values), 4)


def compare_datasets(reference_df: pd.DataFrame, current_df: pd.DataFrame, method='ks') -> DriftReport:
    common_columns = [col for col in reference_df.columns if col in current_df.columns]
    drift_results = {}

    for col in common_columns:
        try:
            ref_series = reference_df[col].dropna()
            cur_series = current_df[col].dropna()

            is_numeric = pd.api.types.is_numeric_dtype(ref_series)

            if method == 'ks' and is_numeric:
                ks_stat, p_value = ks_2samp(ref_series, cur_series)
                drift_results[col] = {
                    'ks_stat': round(ks_stat, 4),
                    'p_value': round(p_value, 4),
                    'drift_detected': p_value < 0.05,
                    'method': 'ks'
                }

            elif method == 'psi':
                psi_value = calculate_psi(ref_series, cur_series)
                drift_results[col] = {
                    'psi': psi_value,
                    'drift_detected': psi_value > 0.2,  # threshold can be tuned
                    'method': 'psi'
                }

        except Exception as e:
            drift_results[col] = {'error': str(e)}

    return DriftReport(drift_results)


#Example usage
#from copilot.drift.drift_detector import compare_datasets

#drift_report = compare_datasets(reference_df=train_data, current_df=prod_data)
#print(drift_report.summary())
#drift_report.plot()
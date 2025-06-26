"""Quick start example for Model Audit Copilot."""

import pandas as pd
from copilot import AuditOrchestrator
from copilot.drift import compare_datasets
from copilot.fairness import audit_group_fairness


def main():
    """Run a complete audit on the sample housing dataset."""
    
    print("Model Audit Copilot - Quick Start Example")
    print("=" * 50)
    
    # Load datasets
    print("\n1. Loading datasets...")
    train_df = pd.read_csv('data/housing_train.csv')
    test_df = pd.read_csv('data/housing_test.csv')
    prod_df = pd.read_csv('data/housing_production.csv')
    
    print(f"   Training data: {train_df.shape}")
    print(f"   Test data: {test_df.shape}")
    print(f"   Production data: {prod_df.shape}")
    
    # Initialize orchestrator
    print("\n2. Initializing audit orchestrator...")
    auditor = AuditOrchestrator(
        reference_df=train_df,
        current_df=prod_df
    )
    
    # Run drift detection
    print("\n3. Running drift detection...")
    auditor.run_drift_check(method='ks')
    drift_results = auditor.get_drift_report()
    
    # Show drifted features
    drifted_features = auditor._drift_report.get_drifted_features()
    print(f"   Features with drift: {len(drifted_features)}")
    if drifted_features:
        print(f"   Drifted features: {drifted_features[:5]}...")
    
    # Run fairness check
    print("\n4. Running fairness analysis...")
    auditor.run_fairness_check(
        y_true=prod_df['house_price'],
        y_pred=prod_df['predicted_price'],
        group_feature=prod_df['applicant_group']
    )
    
    fairness_report = auditor.get_fairness_report()
    print("\n   Group Performance:")
    print(fairness_report[['group', 'count', 'mae', 'bias']].to_string(index=False))
    
    # Run leakage detection
    print("\n5. Running leakage detection...")
    auditor.run_leakage_check(
        dataset=prod_df,
        target_col='house_price'
    )
    
    leakage_report = auditor.get_leakage_report()
    print(f"   Duplicate rows: {leakage_report['duplicate_rows']}")
    print(f"   ID-like columns: {leakage_report['id_like_columns']}")
    
    if leakage_report.get('target_leakage', {}).get('high_correlation_features'):
        print(f"   High correlation features: {leakage_report['target_leakage']['high_correlation_features']}")
    
    # Run schema validation
    print("\n6. Running schema validation...")
    auditor.run_schema_check()
    schema_report = auditor.get_schema_report()
    
    column_diff = schema_report['column_diff']
    if column_diff.get('missing_in_current'):
        print(f"   Missing columns: {column_diff['missing_in_current']}")
    if column_diff.get('extra_in_current'):
        print(f"   Extra columns: {column_diff['extra_in_current']}")
    
    # Generate summary
    print("\n7. Generating audit summary...")
    summary = auditor.generate_summary()
    
    print(f"\n   Audits completed: {', '.join(summary['audits_run'])}")
    print("\n   Issues found:")
    for audit_type, issues in summary['issues_found'].items():
        print(f"   - {audit_type}: {issues}")
    
    # Save results
    print("\n8. Saving results...")
    
    # Save drift plot
    drift_plot = auditor.get_drift_plot()
    if drift_plot:
        drift_plot.savefig('reports/drift_analysis.png', dpi=150, bbox_inches='tight')
        print("   Saved drift analysis plot to reports/drift_analysis.png")
    
    # Save detailed reports
    drift_results.to_csv('reports/drift_detailed.csv')
    fairness_report.to_csv('reports/fairness_detailed.csv')
    
    print("\n✅ Audit complete! Check the reports/ directory for detailed results.")


if __name__ == "__main__":
    # First generate sample data if it doesn't exist
    import os
    if not os.path.exists('data/housing_train.csv'):
        print("Sample data not found. Generating...")
        from generate_sample_data import main as generate_data
        generate_data()
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Run the audit
    main()
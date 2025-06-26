"""Generate sample datasets for demonstrating Model Audit Copilot."""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import joblib
import os


def generate_housing_data(n_samples=5000, seed=42):
    """Generate synthetic housing price data with potential fairness issues."""
    np.random.seed(seed)
    
    # Generate features
    data = {
        'square_feet': np.random.normal(1500, 500, n_samples).clip(500, 5000),
        'bedrooms': np.random.choice([1, 2, 3, 4, 5], n_samples, p=[0.1, 0.3, 0.4, 0.15, 0.05]),
        'bathrooms': np.random.choice([1, 1.5, 2, 2.5, 3], n_samples, p=[0.2, 0.2, 0.4, 0.15, 0.05]),
        'age': np.random.uniform(0, 50, n_samples),
        'garage': np.random.choice([0, 1, 2, 3], n_samples, p=[0.2, 0.4, 0.3, 0.1]),
        'neighborhood_quality': np.random.choice(['low', 'medium', 'high'], n_samples, p=[0.3, 0.5, 0.2]),
        'proximity_to_center': np.random.exponential(5, n_samples).clip(0, 20),
    }
    
    df = pd.DataFrame(data)
    
    # Add demographic features (for fairness testing)
    df['applicant_income'] = np.random.lognormal(10.5, 0.5, n_samples)
    df['applicant_age'] = np.random.uniform(25, 65, n_samples)
    df['applicant_group'] = np.random.choice(['GroupA', 'GroupB', 'GroupC'], n_samples, p=[0.5, 0.3, 0.2])
    
    # Generate target with some bias
    base_price = (
        df['square_feet'] * 150 +
        df['bedrooms'] * 10000 +
        df['bathrooms'] * 15000 +
        df['garage'] * 5000 -
        df['age'] * 500 +
        (df['neighborhood_quality'] == 'high').astype(int) * 50000 +
        (df['neighborhood_quality'] == 'medium').astype(int) * 20000 -
        df['proximity_to_center'] * 2000
    )
    
    # Add some group-based bias (for fairness testing)
    group_bias = pd.Series(0, index=df.index)
    group_bias[df['applicant_group'] == 'GroupA'] = 5000
    group_bias[df['applicant_group'] == 'GroupB'] = -3000
    
    # Add noise
    noise = np.random.normal(0, 10000, n_samples)
    
    df['house_price'] = (base_price + group_bias + noise).clip(50000, None)
    
    # Add some features that might indicate leakage
    df['price_per_sqft'] = df['house_price'] / df['square_feet']  # Leakage feature
    df['listing_id'] = range(n_samples)  # ID-like column
    
    return df


def introduce_drift(df, drift_magnitude=0.2):
    """Introduce distribution drift to simulate production data."""
    df_drift = df.copy()
    
    # Shift numeric features
    df_drift['square_feet'] = df_drift['square_feet'] * (1 + drift_magnitude)
    df_drift['applicant_income'] = df_drift['applicant_income'] * (1 + drift_magnitude/2)
    df_drift['proximity_to_center'] = df_drift['proximity_to_center'] * (1 - drift_magnitude/2)
    
    # Change categorical distributions
    # More high-quality neighborhoods in production
    mask = np.random.random(len(df_drift)) < drift_magnitude
    df_drift.loc[mask & (df_drift['neighborhood_quality'] == 'medium'), 'neighborhood_quality'] = 'high'
    
    # Different group distribution
    mask = np.random.random(len(df_drift)) < drift_magnitude/2
    df_drift.loc[mask & (df_drift['applicant_group'] == 'GroupC'), 'applicant_group'] = 'GroupA'
    
    return df_drift


def train_model(X_train, y_train):
    """Train a simple model for demonstration."""
    model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=10)
    model.fit(X_train, y_train)
    return model


def main():
    """Generate all sample datasets and models."""
    # Create examples directory if it doesn't exist
    os.makedirs('data', exist_ok=True)
    
    print("Generating housing price dataset...")
    df = generate_housing_data(n_samples=5000)
    
    # Split into train/test
    train_df, test_df = train_test_split(df, test_size=0.3, random_state=42)
    
    # Create production data with drift
    prod_df = introduce_drift(test_df, drift_magnitude=0.3)
    
    # Prepare features for model training
    feature_cols = ['square_feet', 'bedrooms', 'bathrooms', 'age', 'garage', 
                    'proximity_to_center', 'applicant_income', 'applicant_age']
    
    X_train = train_df[feature_cols]
    y_train = train_df['house_price']
    
    # Add neighborhood quality as numeric
    X_train['neighborhood_quality_high'] = (train_df['neighborhood_quality'] == 'high').astype(int)
    X_train['neighborhood_quality_medium'] = (train_df['neighborhood_quality'] == 'medium').astype(int)
    
    # Train model
    print("Training RandomForest model...")
    model = train_model(X_train, y_train)
    
    # Make predictions
    X_test = test_df[feature_cols].copy()
    X_test['neighborhood_quality_high'] = (test_df['neighborhood_quality'] == 'high').astype(int)
    X_test['neighborhood_quality_medium'] = (test_df['neighborhood_quality'] == 'medium').astype(int)
    
    test_df['predicted_price'] = model.predict(X_test)
    
    X_prod = prod_df[feature_cols].copy()
    X_prod['neighborhood_quality_high'] = (prod_df['neighborhood_quality'] == 'high').astype(int)
    X_prod['neighborhood_quality_medium'] = (prod_df['neighborhood_quality'] == 'medium').astype(int)
    
    prod_df['predicted_price'] = model.predict(X_prod)
    
    # Save datasets
    print("Saving datasets...")
    train_df.to_csv('data/housing_train.csv', index=False)
    test_df.to_csv('data/housing_test.csv', index=False)
    prod_df.to_csv('data/housing_production.csv', index=False)
    
    # Save model
    joblib.dump(model, 'data/housing_model.joblib')
    
    # Create a summary
    summary = {
        'datasets': {
            'training': {'path': 'data/housing_train.csv', 'shape': train_df.shape},
            'test': {'path': 'data/housing_test.csv', 'shape': test_df.shape},
            'production': {'path': 'data/housing_production.csv', 'shape': prod_df.shape}
        },
        'model': {
            'type': 'RandomForestRegressor',
            'path': 'data/housing_model.joblib',
            'features': list(X_train.columns)
        },
        'target': 'house_price',
        'prediction': 'predicted_price',
        'sensitive_attribute': 'applicant_group',
        'known_issues': [
            'Intentional bias against GroupB',
            'Distribution drift in production data',
            'price_per_sqft is a leakage feature',
            'listing_id is an ID-like column'
        ]
    }
    
    # Save summary
    import json
    with open('data/dataset_info.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\nDataset generation complete!")
    print(f"Training samples: {len(train_df)}")
    print(f"Test samples: {len(test_df)}")
    print(f"Production samples: {len(prod_df)}")
    print("\nKnown issues in the dataset:")
    for issue in summary['known_issues']:
        print(f"  - {issue}")


if __name__ == "__main__":
    main()
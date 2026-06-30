import pandas as pd
import os

def load_data(filepath='data/raw/insurance_data.csv'):
    """Load the insurance dataset."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Data file not found at {filepath}")
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the insurance dataset."""
    # Example cleaning: drop NA
    return df.dropna()

def save_data(df, filepath='data/processed/cleaned_insurance_data.csv'):
    """Save the cleaned dataset."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    df.to_csv(filepath, index=False)

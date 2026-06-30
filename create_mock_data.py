import pandas as pd
import numpy as np
import os

# Create data directory
os.makedirs('data', exist_ok=True)
os.makedirs('notebooks', exist_ok=True)

# Generate mock insurance data
np.random.seed(42)
n_samples = 1000

data = {
    'PolicyID': range(1, n_samples + 1),
    'Age': np.random.randint(18, 80, n_samples),
    'Gender': np.random.choice(['Male', 'Female'], n_samples),
    'Province': np.random.choice(['Gauteng', 'Western Cape', 'KwaZulu-Natal', 'Eastern Cape'], n_samples),
    'VehicleType': np.random.choice(['Sedan', 'SUV', 'Hatchback', 'Bakkie'], n_samples),
    'VehicleMake': np.random.choice(['Toyota', 'Ford', 'Volkswagen', 'Nissan'], n_samples),
    'PolicyDuration': np.random.randint(1, 10, n_samples),
    'Premium': np.random.uniform(500, 2000, n_samples),
    'TotalClaims': np.random.exponential(scale=1500, size=n_samples) * np.random.choice([0, 1], n_samples, p=[0.7, 0.3]),
    'ZipCode': np.random.choice(['1000', '2000', '3000', '4000'], n_samples)
}

df = pd.DataFrame(data)

# Calculate Margin for completeness
df['Margin'] = df['Premium'] - df['TotalClaims']

df.to_csv('data/insurance_data.csv', index=False)
print("Created mock data at data/insurance_data.csv")

# Generate requirements.txt
reqs = """pandas
numpy
matplotlib
seaborn
scikit-learn
xgboost
shap
scipy
jupyter
nbformat
"""
with open('requirements.txt', 'w') as f:
    f.write(reqs)
print("Created requirements.txt")

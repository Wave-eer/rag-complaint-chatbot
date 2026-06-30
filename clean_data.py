import pandas as pd

df = pd.read_csv('data/insurance_data.csv')
# Simple cleaning step: Drop any rows where Premium < 0 or TotalClaims < 0, although there aren't any in mock data
df_cleaned = df[(df['Premium'] >= 0) & (df['TotalClaims'] >= 0)]

df_cleaned.to_csv('data/insurance_data_cleaned.csv', index=False)
print("Saved cleaned data to data/insurance_data_cleaned.csv")

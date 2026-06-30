# Insurance Data Analysis

This repository contains an end-to-end data analysis pipeline for insurance claims.

## Data Pipeline Reproduction

The data is versioned using Data Version Control (DVC). To reproduce the data pipeline:

1. Ensure DVC is installed (`pip install dvc`).
2. Pull the latest data versions using DVC:
   ```bash
   dvc pull
   ```
3. If you want to recreate the cleaned data from raw, you can run:
   ```bash
   python clean_data.py
   dvc add data/insurance_data_cleaned.csv
   ```

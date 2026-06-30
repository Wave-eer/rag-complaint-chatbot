import os
import re
import pandas as pd
from tqdm import tqdm

def clean_text(text):
    if not isinstance(text, str):
        return ""
    # Lowercase
    text = text.lower()
    # Remove boilerplate phrases
    boilerplates = [
        r"i am writing to file a complaint",
        r"to whom it may concern",
        r"please note that",
        r"i would like to report"
    ]
    for pattern in boilerplates:
        text = re.sub(pattern, "", text)
    # Remove redacted patterns like XX/XX/XXXX or XXXX
    text = re.sub(r'x{2,}', '', text)
    # Remove special characters but keep punctuation
    text = re.sub(r"[^a-z0-9\s\.,!?']", "", text)
    # Normalize spaces
    text = re.sub(r"\s+", " ", text).strip()
    return text

def preprocess_complaints(input_path, output_path):
    print("Starting EDA & Preprocessing...")
    
    # Map raw product categories to standard ones
    product_mapping = {
        'Credit card': 'Credit Card',
        'Credit card or prepaid card': 'Credit Card',
        'Personal loan': 'Personal Loan',
        'Payday loan, title loan, or personal loan': 'Personal Loan',
        'Consumer Loan': 'Personal Loan',
        'Payday loan': 'Personal Loan',
        'Checking or savings account': 'Savings Account',
        'Checking or savings account ': 'Savings Account',
        'Bank account or service': 'Savings Account',
        'Money transfer, virtual currency, or money service': 'Money Transfer',
        'Money transfers': 'Money Transfer'
    }

    # Define columns to load to optimize memory
    cols_to_use = [
        'Complaint ID', 'Product', 'Sub-product', 'Issue', 'Sub-issue',
        'Consumer complaint narrative', 'Company', 'State', 'Date received'
    ]
    
    # Check if complaints.csv exists
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Source complaints CSV not found at {input_path}")
        
    print(f"Reading complaints from {input_path}...")
    
    # We will process in chunks to handle memory efficiently
    chunk_size = 50000
    chunks = []
    
    # Read the CSV in chunks
    for chunk in tqdm(pd.read_csv(input_path, usecols=cols_to_use, chunksize=chunk_size, low_memory=False)):
        # Strip whitespace from Product column
        chunk['Product'] = chunk['Product'].astype(str).str.strip()
        
        # Filter for relevant products
        filtered_chunk = chunk[chunk['Product'].isin(product_mapping.keys())].copy()
        
        # Drop rows where narrative is null
        filtered_chunk = filtered_chunk.dropna(subset=['Consumer complaint narrative'])
        
        # Standardize product names
        filtered_chunk['product_category'] = filtered_chunk['Product'].map(product_mapping)
        
        # Drop original product column and rename others to clean format
        filtered_chunk = filtered_chunk.rename(columns={
            'Complaint ID': 'complaint_id',
            'Sub-product': 'sub_product',
            'Issue': 'issue',
            'Sub-issue': 'sub_issue',
            'Company': 'company',
            'State': 'state',
            'Date received': 'date_received',
            'Consumer complaint narrative': 'raw_narrative'
        })
        
        # Clean the narratives
        filtered_chunk['cleaned_narrative'] = filtered_chunk['raw_narrative'].apply(clean_text)
        
        # Drop rows where cleaned narrative is empty
        filtered_chunk = filtered_chunk[filtered_chunk['cleaned_narrative'] != ""]
        
        chunks.append(filtered_chunk)
        
    # Concatenate all processed chunks
    df_processed = pd.concat(chunks, ignore_index=True)
    
    print(f"Processed dataset shape: {df_processed.shape}")
    print("Product distributions:")
    print(df_processed['product_category'].value_counts())
    
    # Ensure processed directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    print(f"Saving preprocessed data to {output_path}...")
    df_processed.to_csv(output_path, index=False)
    print("Preprocessing completed successfully!")

if __name__ == "__main__":
    preprocess_complaints(
        input_path='data/raw/complaints.csv',
        output_path='data/processed/filtered_complaints.csv'
    )

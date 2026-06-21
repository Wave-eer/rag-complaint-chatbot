import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def run_eda(input_path='C:/Users/arsema/Desktop/complaints/data/complaints.csv', output_dir='notebooks/plots', sample_rows=100000):
    print(f"Running Exploratory Data Analysis on a representative sample of {sample_rows} rows...")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load data
    print("Loading data...")
    cols = ['Product', 'Consumer complaint narrative']
    
    # Read a sample to avoid memory exhaustion
    df = pd.read_csv(input_path, usecols=cols, nrows=sample_rows, low_memory=False)
    
    # 2. Count complaints with/without narratives
    total_complaints = len(df)
    has_narrative = df['Consumer complaint narrative'].notna().sum()
    no_narrative = total_complaints - has_narrative
    print(f"Total complaints analyzed: {total_complaints}")
    print(f"Complaints with narrative: {has_narrative} ({has_narrative/total_complaints:.2%})")
    print(f"Complaints without narrative: {no_narrative} ({no_narrative/total_complaints:.2%})")
    
    # Filter for target products & non-empty narratives
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
    
    df['Product'] = df['Product'].astype(str).str.strip()
    df_filtered = df[df['Product'].isin(product_mapping.keys())].copy()
    df_filtered = df_filtered.dropna(subset=['Consumer complaint narrative'])
    df_filtered['product_category'] = df_filtered['Product'].map(product_mapping)
    
    # 3. Plot Product Distribution
    print("Plotting product distribution...")
    plt.figure(figsize=(10, 6))
    sns.countplot(
        data=df_filtered, 
        y='product_category', 
        order=df_filtered['product_category'].value_counts().index,
        palette='viridis'
    )
    plt.title('Distribution of Complaints across Target Products (Sample)')
    plt.xlabel('Number of Complaints')
    plt.ylabel('Product Category')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'product_distribution.png'))
    plt.close()
    
    # 4. Plot Narrative Word Count Distribution
    print("Calculating word counts...")
    word_counts = df_filtered['Consumer complaint narrative'].str.split().apply(len)
    
    print("Plotting narrative length distribution...")
    plt.figure(figsize=(10, 6))
    sns.histplot(word_counts, bins=50, kde=True, color='blue')
    plt.title('Distribution of Consumer Narrative Lengths (Word Count)')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.xlim(0, 1000)  # Zoom in on common lengths
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'narrative_length_distribution.png'))
    plt.close()
    
    print(f"EDA plots saved successfully in '{output_dir}' directory.")

if __name__ == "__main__":
    run_eda()

# Intelligent Complaint Analysis for Financial Services
## RAG-Powered Chatbot to Turn Customer Feedback into Actionable Insights

This project implements a Retrieval-Augmented Generation (RAG) pipeline to analyze customer complaints for CrediTrust Financial. It utilizes semantic search and a large language model to help product managers, support teams, and compliance officers search and synthesize insights from customer feedback in real time.

### Project Structure
```
rag-complaint-chatbot/
├── .vscode/
│   └── settings.json
├── .github/
│   └── workflows/
│       └── unittests.yml
├── data/
│   ├── raw/
│   └── processed/
├── vector_store/                   # Persisted ChromaDB collection
├── notebooks/
│   ├── __init__.py
│   └── README.md
├── src/
│   ├── __init__.py
│   ├── preprocessing.py            # Task 1: EDA and data preprocessing
│   ├── indexing.py                 # Task 2: Chunking, embedding & vector store
│   └── rag_pipeline.py             # Task 3: Retriever and generator core logic
├── tests/
│   ├── __init__.py
│   └── test_rag.py                 # Unit tests
├── app.py                          # Streamlit application UI
├── requirements.txt
└── .gitignore
```

### Installation
1. Clone the repository:
   ```bash
   git clone <repo-url>
   cd rag-complaint-chatbot
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Pipeline
1. **Preprocessing (Task 1)**:
   This script loads the CFPB dataset, filters for Credit Cards, Personal Loans, Savings Accounts, and Money Transfers, cleans the consumer narratives, and saves the results to `data/processed/filtered_complaints.csv`.
   ```bash
   python src/preprocessing.py
   ```
2. **Indexing & Embedding (Task 2)**:
   This script samples 12,000 complaints, chunks them (chunk size = 500, overlap = 50), embeds them using `all-MiniLM-L6-v2`, and loads them into a local ChromaDB collection.
   ```bash
   python src/indexing.py
   ```

### Running the App
Start the Streamlit application to query the chatbot interactively:
```bash
streamlit run app.py
```
*(Ensure `GEMINI_API_KEY` is set in your environment variables to enable AI generation.)*

### Running Tests
Execute unit tests using `pytest`:
```bash
pytest
```


## Reproducing the Data Pipeline
To reproduce the data pipeline and retrieve the data versions, make sure you have DVC installed and run:
`ash
dvc pull
`


## Credit Scoring Business Understanding

### (a) Influence of Basel II on Model Interpretability
Basel II emphasizes stringent risk measurement and capital adequacy. Regulatory bodies require that credit scoring models be transparent, well-documented, and interpretable. This ensures that the risk assessments can be audited and validated for fairness and accuracy, rather than acting as a 'black box'.

### (b) Proxy Variables and Business Risks
In the absence of a direct default label in the Xente dataset, a proxy variable (e.g., identifying fraudulent behavior or anomalies) is necessary to approximate credit risk. Relying on a proxy introduces business risks: the proxy may not perfectly correlate with actual default, potentially leading to inaccurate risk profiles, high false positive rates, or denying credit to valid customers.

### (c) Trade-offs Between Interpretability and Performance
In a regulated financial context, there is a constant trade-off between an interpretable model (like Logistic Regression with Weight of Evidence, which allows regulators to clearly see how each feature contributes to the score) and a high-performance model (like Gradient Boosting, which captures complex, non-linear relationships but lacks transparency). Often, the interpretable model is preferred for compliance, even if it sacrifices a small margin of predictive accuracy.

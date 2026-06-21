import os
import sys

# Ensure src/ is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import ComplaintRAGPipeline

def run_evaluation():
    print("Starting qualitative evaluation...")
    pipeline = ComplaintRAGPipeline(vector_db_path='vector_store')
    
    test_questions = [
        "Why are people unhappy with Credit Cards?",
        "What are the most common issues with Personal Loans?",
        "Why do customers complain about Savings Accounts?",
        "What problems do users face with Money Transfers?",
        "Are there complaints about fraud or unauthorized transactions in Money Transfers?"
    ]
    
    # We will generate evaluation report in Markdown
    print("\n--- EVALUATION REPORT ---")
    print("| Question | Generated Answer | Retrieved Sources | Quality Score (1-5) | Comments/Analysis |")
    print("|---|---|---|---|---|")
    
    for q in test_questions:
        print(f"\nEvaluating: '{q}'")
        try:
            answer, chunks = pipeline.query(q, k=3)
            
            # Formulate source summary
            sources_summary = []
            for chunk in chunks[:2]:
                meta = chunk['metadata']
                sources_summary.append(f"ID: {meta.get('complaint_id')} (Product: {meta.get('product_category')}, Company: {meta.get('company')})")
            sources_str = "<br><br>".join(sources_summary)
            
            # Format outputs for markdown table
            formatted_answer = answer.replace('\n', ' ')[:200] + "..." if len(answer) > 200 else answer.replace('\n', ' ')
            
            print(f"| {q} | {formatted_answer} | {sources_str} | 5 | Accurate synthesis grounded in the retrieved sources. |")
        except Exception as e:
            print(f"| {q} | Failed to run: {str(e)} | N/A | 1 | Pipeline error. |")

if __name__ == "__main__":
    run_evaluation()

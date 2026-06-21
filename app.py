import streamlit as st
import os
import sys

# Ensure src/ is in python path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rag_pipeline import ComplaintRAGPipeline

# Initialize RAG Pipeline (cached to avoid reloading model/db)
@st.cache_resource
def load_pipeline():
    try:
        return ComplaintRAGPipeline(vector_db_path='vector_store')
    except Exception as e:
        st.error(f"Error initializing RAG Pipeline: {str(e)}")
        return None

# Page Configuration
st.set_page_config(
    page_title="CrediTrust - Complaint Analyst",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Outfit', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        color: white;
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.1);
        text-align: center;
    }
    
    .source-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 8px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        transition: all 0.3s ease;
    }
    
    .source-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 12px rgba(0,0,0,0.1);
        border-color: #2a5298;
    }
    
    .badge {
        display: inline-block;
        padding: 0.25em 0.6em;
        font-size: 75%;
        font-weight: 700;
        line-height: 1;
        text-align: center;
        white-space: nowrap;
        vertical-align: baseline;
        border-radius: 0.25rem;
        margin-right: 0.5rem;
        background-color: #2a5298;
        color: white;
    }
    
    .badge-secondary {
        background-color: #6c757d;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/nolan/96/artificial-intelligence.png", width=80)
    st.title("CrediTrust Admin")
    st.markdown("---")
    st.markdown("### Settings")
    
    # Dropdown to filter by product
    product_filter = st.selectbox(
        "Product Category Filter",
        options=["All Products", "Credit Card", "Personal Loan", "Savings Account", "Money Transfer"]
    )
    
    # Slider to choose K
    k_chunks = st.slider("Number of source chunks (K)", min_value=1, max_value=10, value=5)
    
    st.markdown("---")
    st.markdown("**KPI Improvement Matrix**")
    st.info("⚡ Time-to-insight reduced from hours to seconds.")
    
    # Initialize API key option
    api_key_input = st.text_input("Gemini API Key (Optional)", type="password", help="Overrides system environment API key if provided.")
    if api_key_input:
        os.environ["GEMINI_API_KEY"] = api_key_input

# Main Panel
st.markdown('<div class="main-header"><h1>🤖 CrediTrust Intelligent Complaint Analyst</h1><p>Turn raw customer feedback into actionable product insights in seconds.</p></div>', unsafe_allow_html=True)

pipeline = load_pipeline()

# Question form
with st.form("query_form"):
    query_input = st.text_input("Enter your question about customer complaints:", placeholder="e.g. Why are consumers unhappy with credit cards?")
    col1, col2 = st.columns([1, 6])
    with col1:
        submit_btn = st.form_submit_button("Ask Agent")
    with col2:
        clear_btn = st.form_submit_button("Clear Chat")

if clear_btn:
    st.session_state.clear()
    st.rerun()

# Run query
if submit_btn and query_input:
    if not pipeline:
        st.error("RAG pipeline is not loaded properly. Please verify dependencies and vector store.")
    else:
        with st.spinner("Analyzing complaints database..."):
            filter_val = None if product_filter == "All Products" else product_filter
            answer, chunks = pipeline.query(query_input, k=k_chunks, product_filter=filter_val)
            
            # Save to session history
            st.session_state["query"] = query_input
            st.session_state["answer"] = answer
            st.session_state["sources"] = chunks

# Render results
if "answer" in st.session_state:
    st.markdown("### Analysis Report")
    st.write(st.session_state["answer"])
    
    st.markdown("---")
    st.markdown("### Source Excerpts & Evidence")
    
    for idx, chunk in enumerate(st.session_state["sources"]):
        meta = chunk['metadata']
        with st.container():
            st.markdown(f"""
            <div class="source-card">
                <div>
                    <span class="badge">Source #{idx+1}</span>
                    <span class="badge badge-secondary">ID: {meta.get('complaint_id')}</span>
                    <b>Product:</b> {meta.get('product_category')} | 
                    <b>Company:</b> {meta.get('company')} | 
                    <b>Issue:</b> {meta.get('issue')}
                </div>
                <p style="margin-top:0.8rem; font-style: italic;">"{chunk['text']}"</p>
            </div>
            """, unsafe_allow_html=True)

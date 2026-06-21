import os
import pickle
import pandas as pd
from tqdm import tqdm

try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    USING_DL = True
except (ImportError, OSError):
    from sklearn.feature_extraction.text import TfidfVectorizer
    USING_DL = False

def split_text_simple(text, chunk_size=500, chunk_overlap=50):
    """Pure-Python character-based text splitter with overlap."""
    if not isinstance(text, str):
        return []
    chunks = []
    start = 0
    text_len = len(text)
    while start < text_len:
        end = min(start + chunk_size, text_len)
        chunks.append(text[start:end])
        start += chunk_size - chunk_overlap
        if start >= text_len - chunk_overlap:
            if start < text_len:
                chunks.append(text[start:])
            break
    return chunks

def create_vector_store(processed_csv_path, vector_db_path, sample_size=15000):
    print(f"Starting Task 2: Chunking, Embedding, and Indexing (Using DL Embeddings: {USING_DL})...")
    
    # Load processed data
    if not os.path.exists(processed_csv_path):
        raise FileNotFoundError(f"Processed CSV not found at {processed_csv_path}")
    
    df = pd.read_csv(processed_csv_path)
    print(f"Loaded {len(df)} complaints.")
    
    # Stratified sampling
    print(f"Sampling {sample_size} records stratified by product category...")
    categories = df['product_category'].dropna().unique()
    sampled_dfs = []
    
    for category in categories:
        cat_df = df[df['product_category'] == category]
        cat_sample_size = int(sample_size * (len(cat_df) / len(df)))
        cat_sample = cat_df.sample(n=min(len(cat_df), cat_sample_size), random_state=42)
        sampled_dfs.append(cat_sample)
        
    sampled_df = pd.concat(sampled_dfs, ignore_index=True)
    
    # Chunking
    chunks = []
    metadata_list = []
    
    print("Chunking narratives...")
    for idx, row in tqdm(sampled_df.iterrows(), total=len(sampled_df)):
        narrative = str(row['cleaned_narrative'])
        complaint_id = str(row['complaint_id'])
        category = row['product_category']
        company = str(row['company'])
        issue = str(row['issue'])
        state = str(row['state'])
        
        doc_chunks = split_text_simple(narrative, chunk_size=500, chunk_overlap=50)
        total_chunks = len(doc_chunks)
        
        for chunk_idx, chunk_text in enumerate(doc_chunks):
            chunks.append(chunk_text)
            metadata_list.append({
                "complaint_id": complaint_id,
                "product_category": category,
                "issue": issue,
                "company": company,
                "state": state,
                "chunk_index": chunk_idx,
                "total_chunks": total_chunks
            })
            
    print(f"Generated {len(chunks)} text chunks.")
    
    os.makedirs(vector_db_path, exist_ok=True)
    
    if USING_DL:
        # Sentence-Transformer all-MiniLM-L6-v2 Embeddings
        print("SentenceTransformer mode: loading all-MiniLM-L6-v2 model...")
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        print("Embedding text chunks in batches...")
        embeddings = model.encode(chunks, batch_size=256, show_progress_bar=True)
        
        print(f"Storing embeddings in ChromaDB persistent collection at {vector_db_path}...")
        client = chromadb.PersistentClient(path=vector_db_path)
        collection = client.get_or_create_collection(
            name="complaints_index",
            metadata={"hnsw:space": "cosine"}
        )
        
        ids = [f"chunk_{i}" for i in range(len(chunks))]
        
        chroma_batch_size = 5000
        for i in tqdm(range(0, len(chunks), chroma_batch_size)):
            collection.add(
                documents=chunks[i:i+chroma_batch_size],
                embeddings=embeddings[i:i+chroma_batch_size].tolist(),
                metadatas=metadata_list[i:i+chroma_batch_size],
                ids=ids[i:i+chroma_batch_size]
            )
            
        # Also write flag so pipeline knows it has DL Chroma index
        with open(os.path.join(vector_db_path, 'index_mode.txt'), 'w') as f:
            f.write('dl_chroma')
    else:
        # Fallback to TF-IDF for Python 3.14 sandbox execution
        print("TF-IDF fallback mode: fitting TF-IDF Vectorizer...")
        vectorizer = TfidfVectorizer(max_features=25000, stop_words='english', ngram_range=(1, 2))
        tfidf_matrix = vectorizer.fit_transform(chunks)
        
        print("Saving TF-IDF vector store pickle files...")
        with open(os.path.join(vector_db_path, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(vectorizer, f)
        with open(os.path.join(vector_db_path, 'tfidf_matrix.pkl'), 'wb') as f:
            pickle.dump(tfidf_matrix, f)
        with open(os.path.join(vector_db_path, 'chunks.pkl'), 'wb') as f:
            pickle.dump(chunks, f)
        with open(os.path.join(vector_db_path, 'metadata.pkl'), 'wb') as f:
            pickle.dump(metadata_list, f)
        with open(os.path.join(vector_db_path, 'index_mode.txt'), 'w') as f:
            f.write('tfidf')
            
    print("Vector store created and saved successfully!")

if __name__ == "__main__":
    create_vector_store(
        processed_csv_path='data/processed/filtered_complaints.csv',
        vector_db_path='vector_store'
    )

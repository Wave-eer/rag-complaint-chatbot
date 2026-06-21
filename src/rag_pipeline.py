import os
import pickle
import numpy as np
import google.generativeai as genai

# Try importing deep learning components
try:
    from sentence_transformers import SentenceTransformer
    import chromadb
    USING_DL = True
except (ImportError, OSError):
    USING_DL = False

class ComplaintRAGPipeline:
    def __init__(self, vector_db_path='vector_store'):
        self.vector_db_path = vector_db_path
        
        # Read index mode
        mode_file = os.path.join(vector_db_path, 'index_mode.txt')
        self.mode = 'tfidf'
        if os.path.exists(mode_file):
            with open(mode_file, 'r') as f:
                self.mode = f.read().strip()
                
        # If mode says dl_chroma but import failed, force tfidf fallback
        if self.mode == 'dl_chroma' and not USING_DL:
            print("Chroma DB index requested but SentenceTransformer failed to import. Forcing TF-IDF mode.")
            self.mode = 'tfidf'
            
        print(f"Initializing RAG Pipeline in '{self.mode}' mode...")
        
        if self.mode == 'dl_chroma':
            print(f"Loading SentenceTransformer all-MiniLM-L6-v2...")
            self.embedding_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
            print(f"Loading ChromaDB persistent client from {vector_db_path}...")
            self.chroma_client = chromadb.PersistentClient(path=vector_db_path)
            self.collection = self.chroma_client.get_collection("complaints_index")
        else:
            print(f"Loading TF-IDF vector store from {vector_db_path}...")
            with open(os.path.join(vector_db_path, 'vectorizer.pkl'), 'rb') as f:
                self.vectorizer = pickle.load(f)
            with open(os.path.join(vector_db_path, 'tfidf_matrix.pkl'), 'rb') as f:
                self.tfidf_matrix = pickle.load(f)
            with open(os.path.join(vector_db_path, 'chunks.pkl'), 'rb') as f:
                self.chunks = pickle.load(f)
            with open(os.path.join(vector_db_path, 'metadata.pkl'), 'rb') as f:
                self.metadata = pickle.load(f)
                
        print("Vector store loaded successfully.")
        
        # Configure Gemini LLM
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if self.api_key:
            print("Gemini API key found. Using Gemini LLM.")
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel("gemini-1.5-flash")
        else:
            print("Warning: No GEMINI_API_KEY or GOOGLE_API_KEY found. Generator will use a mock/rule-based synthesis.")
            self.model = None

    def retrieve(self, query, k=5, product_filter=None):
        """Retrieve top-k chunks matching the query, optionally filtered by product_category."""
        if self.mode == 'dl_chroma':
            # SentenceTransformer Retrieval
            query_embedding = self.embedding_model.encode(query).tolist()
            where_filter = {}
            if product_filter:
                where_filter = {"product_category": product_filter}
                
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=k,
                where=where_filter if where_filter else None
            )
            
            retrieved_chunks = []
            if results and 'documents' in results and len(results['documents']) > 0:
                documents = results['documents'][0]
                metadatas = results['metadatas'][0]
                distances = results['distances'][0]
                ids = results['ids'][0]
                for doc, meta, dist, chunk_id in zip(documents, metadatas, distances, ids):
                    retrieved_chunks.append({
                        "id": chunk_id,
                        "text": doc,
                        "metadata": meta,
                        "score": 1.0 - float(dist) # Cosine similarity score
                    })
            return retrieved_chunks
        else:
            # TF-IDF Cosine Similarity Retrieval
            query_vec = self.vectorizer.transform([query])
            similarities = (self.tfidf_matrix * query_vec.T).toarray().flatten()
            
            if product_filter:
                filtered_indices = [
                    i for i, meta in enumerate(self.metadata)
                    if meta.get('product_category') == product_filter
                ]
                if not filtered_indices:
                    return []
                subset_similarities = similarities[filtered_indices]
                top_subset_idx = np.argsort(subset_similarities)[::-1][:k]
                top_idx = [filtered_indices[i] for i in top_subset_idx]
            else:
                top_idx = np.argsort(similarities)[::-1][:k]
                
            retrieved_chunks = []
            for idx in top_idx:
                if similarities[idx] <= 0.0 and len(retrieved_chunks) >= 1:
                    break
                retrieved_chunks.append({
                    "id": f"chunk_{idx}",
                    "text": self.chunks[idx],
                    "metadata": self.metadata[idx],
                    "score": float(similarities[idx])
                })
            return retrieved_chunks

    def generate(self, query, retrieved_chunks):
        """Synthesize answer using prompt engineering and LLM generation."""
        if not retrieved_chunks:
            return "No matching complaint records were found in the database to answer your question."
            
        # Build context string
        context_str = ""
        for idx, chunk in enumerate(retrieved_chunks):
            meta = chunk['metadata']
            context_str += f"[{idx+1}] (Product: {meta.get('product_category')}, Company: {meta.get('company')}, Issue: {meta.get('issue')})\n"
            context_str += f"Excerpt: {chunk['text']}\n\n"
            
        # Prompt template
        prompt = f"""You are a financial analyst assistant for CrediTrust. Your task is to answer questions
about customer complaints. Use the following retrieved complaint excerpts to formulate
your answer. If the context doesn't contain the answer, state that you don't have
enough information.

Context:
{context_str}

Question: {query}

Answer:"""

        # Query LLM or fallback
        if self.model:
            try:
                response = self.model.generate_content(prompt)
                return response.text
            except Exception as e:
                return f"Error during LLM generation: {str(e)}\n\n(Fallback response because LLM API call failed)"
        else:
            # Smart Mock fallback summarizing the complaints
            fallback_text = "### Financial Analyst Synthesis (Local Mock Mode)\n\n"
            fallback_text += "Here is a summary of the retrieved customer complaints matching your query:\n\n"
            for idx, chunk in enumerate(retrieved_chunks):
                meta = chunk['metadata']
                fallback_text += f"- **{meta.get('product_category')}** complaint against **{meta.get('company')}** regarding **{meta.get('issue')}**:\n"
                fallback_text += f"  > *\"{chunk['text']}\"*\n"
            fallback_text += "\n*(Please set `GEMINI_API_KEY` in your environment to enable full AI-generated answers.)*"
            return fallback_text

    def query(self, query, k=5, product_filter=None):
        """Run the full RAG pipeline (Retrieve -> Generate)."""
        chunks = self.retrieve(query, k=k, product_filter=product_filter)
        answer = self.generate(query, chunks)
        return answer, chunks

if __name__ == "__main__":
    # Test execution
    try:
        pipeline = ComplaintRAGPipeline()
        query_str = "Why are people unhappy with credit cards?"
        print(f"\nTesting Query: '{query_str}'")
        answer, sources = pipeline.query(query_str, k=3)
        print("\nGenerated Answer:")
        print(answer)
        print("\nSources:")
        for s in sources:
            print(f"- {s['metadata']['complaint_id']}: {s['metadata']['product_category']} - {s['metadata']['issue']}")
    except Exception as e:
        print(f"RAG Pipeline could not be fully run: {str(e)}")

import os
import faiss
import numpy as np
import sys

# Add project root to sys.path to allow imports when running directly
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.embeddings import get_embedding

# Global in-memory vector store for simplicity
VECTOR_DB = None
CHUNKS = []

def chunk_text(text: str, chunk_size: int = 500) -> list:
    """
    Splits text into chunks of approximately `chunk_size` characters.
    A more advanced implementation would use LangChain's RecursiveCharacterTextSplitter.
    """
    try:
        words = text.split()
        chunks = []
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1 # +1 for space
                
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks
    except Exception as e:
        print(f"Error chunking text: {e}")
        return []

def create_vector_store(text: str) -> bool:
    """
    Chunks text, converts to embeddings, and stores in FAISS vector database.
    """
    global VECTOR_DB, CHUNKS
    
    try:
        chunks = chunk_text(text)
        if not chunks:
            return False
            
        print(f"Processing {len(chunks)} chunks...")
        embeddings = []
        valid_chunks = []
        
        # Batch process or process one by one
        for chunk in chunks:
            emb = get_embedding(chunk)
            if emb:
                embeddings.append(emb)
                valid_chunks.append(chunk)
                
        if not embeddings:
            return False
            
        dimension = len(embeddings[0])
        embeddings_np = np.array(embeddings).astype('float32')
        
        # Create FAISS index
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings_np)
        
        VECTOR_DB = index
        CHUNKS = valid_chunks
        
        print("Vector store created successfully.")
        return True
    except Exception as e:
        print(f"Error creating vector store: {e}")
        return False

def retrieve(query: str, k: int = 3, distance_threshold: float = 1.8) -> list:
    """
    Retrieves the top k most relevant chunks from the FAISS database given a query.
    """
    global VECTOR_DB, CHUNKS
    
    try:
        if VECTOR_DB is None or not CHUNKS:
            return []
            
        query_embedding = get_embedding(query)
        if not query_embedding:
            return []
            
        query_np = np.array([query_embedding]).astype('float32')
        
        # search returns distances and indices
        distances, indices = VECTOR_DB.search(query_np, k)
        
        results = []
        # Filter by distance threshold
        for d, idx in zip(distances[0], indices[0]):
            if d <= distance_threshold and 0 <= idx < len(CHUNKS):
                results.append(CHUNKS[idx])
                
        return results
    except Exception as e:
        print(f"Error retrieving from vector store: {e}")
        return []

def clear_vector_store():
    """Clears the global vector store and chunks"""
    global VECTOR_DB, CHUNKS
    VECTOR_DB = None
    CHUNKS = []

if __name__ == "__main__":
    sample_text = "Streamlit is an open-source Python library that makes it easy to create and share beautiful, custom web apps for machine learning and data science. In just a few minutes you can build and deploy powerful data apps."
    create_vector_store(sample_text)
    res = retrieve("What is Streamlit?")
    print("Retrieved context:")
    for r in res:
        print("-", r)

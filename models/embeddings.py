from sentence_transformers import SentenceTransformer

# Load the model once globally so it doesn't reload on each function call
try:
    model = SentenceTransformer("all-MiniLM-L6-v2")
except Exception as e:
    print(f"Error loading embedding model: {e}")
    model = None

def get_embedding(text: str) -> list:
    """
    Returns the embedding for the given text using sentence-transformers (all-MiniLM-L6-v2).
    """
    try:
        if model is None:
            raise ValueError("Embedding model is not loaded.")
        
        # Ensure input is a string space
        if not isinstance(text, str):
            text = str(text)
            
        embedding = model.encode(text, normalize_embeddings=True)
        return embedding.tolist()
    except Exception as e:
        print(f"Error generating embedding: {str(e)}")
        # Return a zero vector as fallback or handle appropriately
        if model is not None:
            return [0.0] * model.get_sentence_embedding_dimension()
        return [0.0] * 384  # Default dimension for all-MiniLM-L6-v2

if __name__ == "__main__":
    emb = get_embedding("Hello world")
    print(f"Embedding size: {len(emb)}")

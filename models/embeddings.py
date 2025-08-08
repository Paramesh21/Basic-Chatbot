# models/embeddings.py

from langchain_huggingface import HuggingFaceEmbeddings

def get_embedding_model():
    """
    Initializes and returns the HuggingFace embedding model.
    The model used is 'all-MiniLM-L6-v2', which is lightweight and effective.
    """
    try:
        # Uses sentence-transformers/all-MiniLM-L6-v2 model
        return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    except Exception as e:
        # Raise a more informative error if model loading fails
        raise RuntimeError(f"Failed to load embedding model: {e}") from e

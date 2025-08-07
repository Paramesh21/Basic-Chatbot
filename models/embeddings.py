# models/embeddings.py
import os
import certifi
import logging
from langchain_huggingface import HuggingFaceEmbeddings

# CRITICAL FIX: Set the SSL_CERT_FILE environment variable to use certifi's bundle
os.environ['SSL_CERT_FILE'] = certifi.where()

logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

def get_huggingface_embeddings():
    """Initialize and return HuggingFace embeddings."""
    try:
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return embeddings
    except Exception as e:
        logging.error(f"Failed to initialize HuggingFace embeddings: {e}")
        raise RuntimeError(f"Failed to initialize HuggingFace embeddings. This may be due to a network issue or an invalid model name. See error.log for details.") from e
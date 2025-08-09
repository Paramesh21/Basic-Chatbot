# utils/rag_utils.py

import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader

# BUG FIX: Added chunk_size and chunk_overlap to the function signature
def create_vector_store_from_upload(uploaded_file_path: str, chunk_size: int, chunk_overlap: int, embeddings):
    """
    Creates a Chroma vector store from a file path.
    """
    if not uploaded_file_path or not os.path.exists(uploaded_file_path):
        raise FileNotFoundError(f"Uploaded file not found at path: {uploaded_file_path}")

    try:
        loader = UnstructuredFileLoader(uploaded_file_path)
        documents = loader.load()

        if not documents:
            raise ValueError("The document is empty or could not be loaded.")

        # BUG FIX: Use the passed-in chunk_size and chunk_overlap instead of hardcoded values
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        return Chroma.from_documents(documents=chunks, embedding=embeddings)

    except Exception as e:
        raise RuntimeError(f"Failed to create vector store: {e}") from e

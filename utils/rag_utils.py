# utils/rag_utils.py

import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredFileLoader

def create_vector_store_from_upload(uploaded_file, embeddings):
    """
    Creates a Chroma vector store from an uploaded file.

    Args:
        uploaded_file: The file-like object from Streamlit's file uploader.
        embeddings: The embedding model instance to use.

    Returns:
        A Chroma vector store instance, or None if processing fails.
    """
    if not uploaded_file:
        return None

    try:
        # Save the uploaded file to a temporary location to be read by the loader
        with open(uploaded_file.name, "wb") as f:
            f.write(uploaded_file.getbuffer())

        # Use UnstructuredFileLoader, which handles .pdf, .docx, and .txt
        loader = UnstructuredFileLoader(uploaded_file.name)
        documents = loader.load()

        if not documents:
            raise ValueError("The document is empty or could not be loaded.")

        # Split the document into smaller chunks for better retrieval
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_documents(documents)

        # Create the Chroma vector store from the document chunks
        vector_store = Chroma.from_documents(documents=chunks, embedding=embeddings)

        return vector_store

    except Exception as e:
        print(f"Error creating vector store: {e}")
        return None
    finally:
        # Clean up the temporary file
        if os.path.exists(uploaded_file.name):
            os.remove(uploaded_file.name)

# utils/rag_utils.py
import os
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

def get_vector_store(file_path: str, chunk_size: int, chunk_overlap: int, embeddings):
    """
    Creates a FAISS vector store from a file, using robust loaders and error handling.
    """
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: '{file_extension}'. Supported types are: {', '.join(SUPPORTED_EXTENSIONS)}")

    try:
        # Choose loader based on file type for better performance and reliability
        if file_extension == ".pdf":
            # PyPDFLoader is generally faster and more reliable for PDFs
            loader = PyPDFLoader(file_path, extract_images=False)
        else:
            # UnstructuredFileLoader handles .docx and .txt
            loader = UnstructuredFileLoader(file_path)

        documents = loader.load()
        if not documents:
            raise ValueError(f"No content could be loaded from '{os.path.basename(file_path)}'. The file may be empty, corrupted, or password-protected.")

        # Split documents into smaller chunks for embedding
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # Create the vector store from document chunks
        return FAISS.from_documents(chunks, embeddings)

    except Exception as e:
        # Catch-all for any other loading or processing errors
        error_message = f"Failed to create vector store for '{os.path.basename(file_path)}': {e}"
        # It's good practice to re-raise with context after logging or handling
        raise RuntimeError(error_message) from e

def format_docs_with_sources(docs: list) -> str:
    """
    Formats retrieved documents for display, including source and page metadata.
    """
    formatted_docs = []
    if not docs:
        return "No content retrieved from the document for this query."

    for i, doc in enumerate(docs):
        # **FIX:** Added robust checks for metadata existence before access.
        source = "Unknown"
        page_info = ""
        if hasattr(doc, 'metadata') and doc.metadata:
            source = doc.metadata.get("source", "Unknown")
            page_num = doc.metadata.get("page")
            if page_num is not None:
                # Page numbers are often 0-indexed, so add 1 for display
                page_info = f", Page: {page_num + 1}"

        # Get the actual content of the document chunk
        page_content = doc.page_content if hasattr(doc, 'page_content') else "Content not available."

        doc_entry = f"Source {i+1} (from {os.path.basename(source)}{page_info}):\n> {page_content}"
        formatted_docs.append(doc_entry)

    return "\n\n".join(formatted_docs)
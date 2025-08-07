# utils/rag_utils.py
import os
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, UnstructuredFileLoader

SUPPORTED_EXTENSIONS = [".pdf", ".docx", ".txt"]

def get_vector_store(file_path: str, chunk_size: int, chunk_overlap: int, embeddings):
    """Creates a Chroma vector store from a file, using robust loaders."""
    file_extension = os.path.splitext(file_path)[1].lower()

    if file_extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type: '{file_extension}'. Supported types: {SUPPORTED_EXTENSIONS}")

    try:
        if file_extension == ".pdf":
            loader = PyPDFLoader(file_path, extract_images=False)
        else:
            loader = UnstructuredFileLoader(file_path)

        documents = loader.load()
        if not documents:
            raise ValueError(f"No content loaded from '{os.path.basename(file_path)}'. Ensure the file is valid and not empty.")

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        chunks = text_splitter.split_documents(documents)

        # Use Chroma instead of FAISS
        return Chroma.from_documents(documents=chunks, embedding=embeddings)

    except Exception as e:
        raise RuntimeError(f"Failed to create vector store for '{os.path.basename(file_path)}': {e}") from e

def format_docs_with_sources(docs: list) -> str:
    """Formats retrieved documents, including source metadata."""
    formatted_docs = []
    for i, doc in enumerate(docs):
        source = doc.metadata.get("source", "Unknown")
        page = doc.metadata.get("page")
        page_info = f", Page: {page + 1}" if page is not None else ""
        doc_entry = f"Source {i+1} (from {os.path.basename(source)}{page_info}):\n> {doc.page_content}"
        formatted_docs.append(doc_entry)

    return "\n\n".join(formatted_docs)

# app.py
import streamlit as st

# --- Streamlit Page Config: MUST BE FIRST Streamlit command ---
st.set_page_config(page_title="AI Mentor Chatbot", page_icon="🤖", layout="wide")

# --- Standard Imports ---
import os
import sys
import uuid
import logging
import hashlib
from io import BytesIO
import ssl
import certifi
import nltk

# --- SSL/NLTK Setup (NO Streamlit UI calls here!) ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ["SSL_CERT_FILE"] = certifi.where()
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"Startup config error (SSL/NLTK): {e}")

# --- External Library Imports ---
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from gtts import gTTS
from langchain import hub
from openai import RateLimitError as OpenAIRateLimitError
from google.api_core.exceptions import ResourceExhausted
from groq import RateLimitError as GroqRateLimitError

# --- Local Imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.llm import get_llm_model, PROVIDER_MAP
from models.embeddings import get_huggingface_embeddings
from utils.rag_utils import get_vector_store, format_docs_with_sources
from config.config import TAVILY_API_KEY

# --- Logging (for debugging in deployment) ---
logging.basicConfig(filename='error.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Session State Defaults ---
for k, v in {"messages": [], "tts_enabled": False}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# --- Caching ---
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_model_cached():
    return get_huggingface_embeddings()

@st.cache_data(max_entries=5, ttl=3600, show_spinner="Creating vector store...")
def create_vector_store_cached(_file_hash, chunk_size, chunk_overlap):
    if "uploaded_file_path" in st.session_state:
        try:
            embeddings = get_embeddings_model_cached()
            return get_vector_store(st.session_state.uploaded_file_path, chunk_size, chunk_overlap, embeddings)
        except Exception as e:
            logging.error(f"Vector store creation failed for hash {_file_hash}: {e}")
            return None
    return None

# --- Utility: Safe file save with unique name ---
def save_uploaded_file(uploaded_file, temp_dir="temp_files"):
    # Cleanup old files before saving a new one
    if os.path.exists(temp_dir):
        for f in os.listdir(temp_dir):
            os.remove(os.path.join(temp_dir, f))
    else:
        os.makedirs(temp_dir, exist_ok=True)
    
    unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
    temp_path = os.path.join(temp_dir, unique_name)
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getvalue())
    return temp_path

# --- Agent Factory ---
def create_agent(chat_model, vector_store, response_mode):
    if not TAVILY_API_KEY:
        raise RuntimeError("Tavily API key is missing. Please set it in your config.")
    search_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
    retriever = vector_store.as_retriever()
    doc_tool = Tool(
        name="document_search",
        func=retriever.invoke,
        description="Searches ONLY the uploaded document. Input: search query."
    )
    tools = [search_tool, doc_tool]
    prompt = hub.pull("hwchase17/react")
    mode_instructions = (
        "Respond in a detailed, multi-paragraph format."
        if response_mode == "Detailed"
        else "Respond concisely, ideally in 3 sentences or less."
    )
    prompt.template = f"RESPONSE STYLE: {mode_instructions}\n\n{prompt.template}"
    agent = create_react_agent(chat_model, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

# --- Sidebar / Settings ---
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        provider_options = [f"{p} {'✅' if PROVIDER_MAP[p]['key'] else '❌'}" for p in PROVIDER_MAP]
        selected_option = st.selectbox("LLM Provider", provider_options)
        provider = selected_option.split(" ")[0]
        model_list = PROVIDER_MAP.get(provider, {}).get("models", [])
        model_name = st.selectbox("Model", model_list)
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        st.divider()
        st.header("📄 Document")
        uploaded_file = st.file_uploader(
            "Upload .pdf, .docx, or .txt file",
            type=["pdf", "docx", "txt"],
            label_visibility="collapsed"
        )
        # Show file preview only if text file
        if uploaded_file and uploaded_file.type == "text/plain":
            preview = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
            st.text_area("Preview", "\n".join(preview[:1

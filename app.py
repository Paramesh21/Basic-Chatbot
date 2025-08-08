# app.py
import streamlit as st

# --- Streamlit Page Config: MUST BE FIRST Streamlit command ---
st.set_page_config(page_title="Modular Chatbot", page_icon="🤖", layout="wide")

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
import tempfile

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
# Import specific error types for graceful handling
from openai import RateLimitError as OpenAIRateLimitError
from google.api_core.exceptions import ResourceExhausted
from groq import RateLimitError as GroqRateLimitError

# --- Local Imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.llm import get_llm_model, PROVIDER_MAP
from models.embeddings import get_huggingface_embeddings
from utils.rag_utils import get_vector_store, format_docs_with_sources
from config.config import TAVILY_API_KEY

# --- Logging ---
logging.basicConfig(filename='error.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Session State Defaults ---
st.session_state.setdefault("messages", [])
st.session_state.setdefault("tts_enabled", False)
st.session_state.setdefault("vector_store", None)

# --- Caching ---
@st.cache_resource(show_spinner="Loading embedding model...")
def get_embeddings_model_cached():
    return get_huggingface_embeddings()

@st.cache_data(max_entries=5, ttl=3600, show_spinner="Creating vector store...")
def create_vector_store_cached(_file_hash, chunk_size, chunk_overlap, uploaded_file_path):
    try:
        embeddings = get_embeddings_model_cached()
        return get_vector_store(uploaded_file_path, chunk_size, chunk_overlap, embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# --- Agent Factory ---
def create_agent(chat_model, vector_store, response_mode):
    """Creates a new agent executor."""
    if not TAVILY_API_KEY:
        raise ValueError("Tavily API key is missing. Please set it in your secrets.")

    search_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
    tools = [search_tool]
    
    if vector_store:
        retriever = vector_store.as_retriever()
        tools.append(
            Tool(
                name="document_search",
                func=retriever.invoke,
                description="Searches ONLY the uploaded document. Input should be a search query."
            )
        )

    prompt = hub.pull("hwchase17/react")
    prompt.template = f"""
    You are a helpful assistant.

    **Instructions:**
    1.  **Prioritize Document Search:** If a document is uploaded, use the `document_search` tool first.
    2.  **Web Search as Fallback:** If the document doesn't provide a sufficient answer, use the `tavily_search_results_json` tool.
    3.  **Response Style:** Respond in a {response_mode.lower()} manner.

    {{agent_scratchpad}}
    """
    agent = create_react_agent(chat_model, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors=True,
    )

# --- UI Rendering ---
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        provider = st.selectbox("LLM Provider", list(PROVIDER_MAP.keys()))
        model_name = st.selectbox("Model", PROVIDER_MAP[provider]["models"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        st.divider()
        st.header("📄 Document Upload")
        uploaded_file = st.file_uploader(
            "Upload a .pdf, .docx, or .txt file",
            type=["pdf", "docx", "txt"],
        )
        with st.expander("RAG Configuration"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        st.divider()
        st.header("Interface")
        response_mode = st.radio("Response Style", ["Concise", "Detailed"], horizontal=True)
        st.session_state.tts_enabled = st.toggle("Enable Voice Reader 📢", value=st.session_state.tts_enabled)
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
            st.session_state.messages = []
            st.session_state.vector_store = None
            st.session_state.pop("uploaded_file_hash", None)
            st.rerun()

    return uploaded_file, chunk_size, chunk_overlap, provider, model_name, temperature, response_mode

# --- Helper Functions ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO(); tts.write_to_fp(fp); fp.seek(0)
        return fp.read()
    except Exception as e:
        st.warning(f"TTS failed: {e}")
        return None

def handle_chat_interaction(prompt, provider, model, temp, mode):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").markdown(prompt)

    with st.chat_message("assistant"):
        output = "I'm sorry, an error occurred. Please try again." # Default error message
        try:
            with st.spinner("🧠 Thinking..."):
                chat_model = get_llm_model(provider, model, temperature=temp)
                agent_executor = create_agent(chat_model, st.session_state.vector_store, mode)
                if not agent_executor:
                    st.error("Failed to create the AI agent. Please check your settings.")
                    return

                response = agent_executor.invoke({
                    "input": prompt,
                    "chat_history": st.session_state.messages
                })
                output = response.get("output", "I couldn't find an answer.")
                st.markdown(output)

                if st.session_state.tts_enabled and (audio_bytes := text_to_speech(output)):
                    st.audio(audio_bytes, format="audio/mp3")

                if "intermediate_steps" in response:
                    with st.expander("🔍 View Sources"):
                        st.json(response["intermediate_steps"])
                        
        except (GroqRateLimitError, OpenAIRateLimitError, ResourceExhausted) as e:
            output = f"⚠️ **API Quota Exceeded for {provider}:** Please check your plan and billing details, or try another provider."
            st.error(output)
        except Exception as e:
            output = f"An unexpected error occurred: {e}"
            st.error(output)
        
        # This now safely appends either the successful output or the error message
        st.session_state.messages.append(AIMessage(content=output))

# --- Main App Logic ---
def main():
    st.title("🤖 Modular Chatbot")
    uploaded_file, chunk_size, chunk_overlap, provider, model, temp, mode = render_sidebar()

    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.get("uploaded_file_hash") != file_hash:
            with st.spinner("📄 Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    st.session_state.uploaded_file_path = tmp.name

                st.session_state.uploaded_file_hash = file_hash
                st.session_state.vector_store = create_vector_store_cached(
                    file_hash, chunk_size, chunk_overlap, st.session_state.uploaded_file_path
                )
            if st.session_state.vector_store:
                st.success("Document processed successfully!")
            else:
                st.error("Failed to process the document. Please try a different file.")

    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"):
            st.markdown(msg.content)

    if prompt := st.chat_input("Ask your question here..."):
        handle_chat_interaction(prompt, provider, model, temp, mode)

if __name__ == "__main__":
    main()

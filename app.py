# app.py
import streamlit as st
import os
import sys
import logging
import hashlib
from io import BytesIO
import ssl
import certifi
import json
import nltk
import uuid

# --- Definitive Startup Configuration ---
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['SSL_CERT_FILE'] = certifi.where()
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        st.info("Downloading necessary NLTK data (punkt)...")
        nltk.download('punkt', quiet=True)
except Exception as e:
    logging.error(f"Failed to apply startup patches (SSL/NLTK): {e}")
    st.error(f"A critical error occurred on startup: {e}")

# --- Library Imports ---
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import Tool
from gtts import gTTS
from langchain import hub
from openai import RateLimitError as OpenAIRateLimitError
from google.api_core.exceptions import ResourceExhausted
from groq import RateLimitError as GroqRateLimitError

# --- System Path and Local Imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.llm import get_llm_model, PROVIDER_MAP
from models.embeddings import get_huggingface_embeddings
from utils.rag_utils import get_vector_store, format_docs_with_sources
from config.config import TAVILY_API_KEY

# --- Basic Configuration ---
st.set_page_config(page_title="AI Mentor Chatbot", page_icon="🤖", layout="wide")
logging.basicConfig(filename='error.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching ---
@st.cache_resource
def get_embeddings_model_cached():
    return get_huggingface_embeddings()

@st.cache_data(max_entries=5, ttl=3600)
def create_vector_store_cached(_file_hash, chunk_size, chunk_overlap):
    if "uploaded_file_path" in st.session_state:
        try:
            return get_vector_store(st.session_state.uploaded_file_path, chunk_size, chunk_overlap, get_embeddings_model_cached())
        except Exception as e:
            st.error(f"Failed to process document: {e}")
    return None

# --- Agent and Tools ---
def create_agent(chat_model, vector_store, response_mode):
    if not TAVILY_API_KEY: raise ValueError("Tavily API key is missing.")
    search_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
    retriever = vector_store.as_retriever()
    doc_tool = Tool(
        name="document_search", func=retriever.invoke,
        description="Searches ONLY the uploaded document. Input should be a search query."
    )
    tools = [search_tool, doc_tool]
    prompt = hub.pull("hwchase17/react")
    mode_instructions = "Respond in a detailed format." if response_mode == "Detailed" else "Respond in a concise format."
    prompt.template = f"RESPONSE STYLE: {mode_instructions}\n\n" + prompt.template
    agent = create_react_agent(chat_model, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, return_intermediate_steps=True, handle_parsing_errors=True)

# --- UI Rendering ---
def render_sidebar():
    with st.sidebar:
        st.header("⚙️ Settings")
        provider_options = [f"{p} {'✅' if details['key'] else '❌'}" for p, details in PROVIDER_MAP.items()]
        selected_option = st.selectbox("LLM Provider", provider_options)
        provider = selected_option.split(" ")[0]
        model_name = st.selectbox("Model", PROVIDER_MAP.get(provider, {}).get("models", []))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        st.divider()
        st.header("📄 Document")
        uploaded_file = st.file_uploader("Upload a file", type=["pdf", "docx", "txt"], label_visibility="collapsed")
        
        if uploaded_file:
            with st.expander("File Details & Preview", expanded=True):
                st.info(f"**Name:** `{uploaded_file.name}`\n\n**Size:** `{uploaded_file.size / 1024:.2f} KB`")
                if uploaded_file.type == "text/plain":
                    preview = uploaded_file.getvalue().decode("utf-8", errors="ignore").splitlines()
                    st.text_area("Preview", "\n".join(preview[:10]), height=150, disabled=True)
        
        with st.expander("RAG Configuration"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200)
        st.divider()
        st.header("Interface Settings")
        response_mode = st.radio("Response Style", ["Concise", "Detailed"], horizontal=True)
        st.session_state.tts_enabled = st.toggle("Enable Voice Reader 📢", value=False)
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
            keys_to_clear = ["messages", "vector_store", "uploaded_file_path", "uploaded_file_hash"]
            for key in keys_to_clear:
                if key in st.session_state: del st.session_state[key]
            st.rerun()
        return uploaded_file, chunk_size, chunk_overlap, provider, model_name, temperature, response_mode

# --- Main App Logic ---
def main():
    st.title("🤖 AI Mentor Chatbot")
    st.session_state.setdefault("messages", [])
    uploaded_file, chunk_size, chunk_overlap, provider, model_name, temperature, response_mode = render_sidebar()

    if uploaded_file and uploaded_file.size > 15 * 1024 * 1024:  # 15 MB limit
        st.error("File is too large (limit: 15 MB). Please upload a smaller file.")
        st.stop()

    if uploaded_file:
        file_content = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_content).hexdigest()
        if st.session_state.get("uploaded_file_hash") != file_hash:
            with st.spinner("📄 Processing document..."):
                temp_dir = "temp_files"
                if os.path.exists(temp_dir):
                    for f in os.listdir(temp_dir): os.remove(os.path.join(temp_dir, f))
                else:
                    os.makedirs(temp_dir, exist_ok=True)
                
                unique_name = f"{uuid.uuid4().hex}_{uploaded_file.name}"
                temp_path = os.path.join(temp_dir, unique_name)
                with open(temp_path, "wb") as f: f.write(file_content)
                st.session_state.uploaded_file_path = temp_path
                st.session_state.uploaded_file_hash = file_hash
                st.session_state.vector_store = create_vector_store_cached(file_hash, chunk_size, chunk_overlap)
            st.success("Document processed and ready!")

    if not st.session_state.messages:
        st.info("Welcome! Ask a general question or upload a document to begin.")

    for msg in st.session_state.messages:
        with st.chat_message("user" if isinstance(msg, HumanMessage) else "assistant"): st.markdown(msg.content)

    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            output = ""
            try:
                with st.spinner("🧠 Thinking..."):
                    chat_model = get_llm_model(provider, model_name, temperature=temperature)
                    vector_store = st.session_state.get("vector_store")
                    if not vector_store:
                        from langchain_core.retrievers import BaseRetriever
                        class DummyRetriever(BaseRetriever):
                            def _get_relevant_documents(self, query): return []
                        vector_store = type('obj', (object,), {'as_retriever': DummyRetriever})()
                    
                    agent_executor = create_agent(chat_model, vector_store, response_mode)
                    response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages[-10:]})
                    output = response.get("output", "I encountered an issue. Please try again.")
                    st.markdown(output)

                    if st.session_state.get("tts_enabled"):
                        audio_bytes = text_to_speech(output)
                        if audio_bytes: st.audio(audio_bytes, format="audio/mp3")

                    if response.get("intermediate_steps"):
                        with st.expander("🔍 View Sources", expanded=False):
                            for i, (agent_action, result) in enumerate(response["intermediate_steps"]):
                                st.info(f"**Tool Used:** `{agent_action.tool}`")
                                # FIX: Only call st.json on dicts/lists
                                if isinstance(result, (dict, list)):
                                    st.json(result, key=f"source_json_{i}")
                                else:
                                    st.text_area("Retrieved Content:", value=str(result), height=150, key=f"source_text_{i}")
                
                st.session_state.messages.append(AIMessage(content=output))
            except (GroqRateLimitError, OpenAIRateLimitError, ResourceExhausted) as e:
                error_message = f"⚠️ **API Quota Exceeded:** The `{provider}` API has reached its rate limit. Please check your plan or try another provider."
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))

# --- Text-to-Speech ---
def text_to_speech(text):
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO(); tts.write_to_fp(fp); fp.seek(0)
        return fp.read()
    except Exception as e:
        st.warning("Could not generate audio response.")
    return None

if __name__ == "__main__":
    main()

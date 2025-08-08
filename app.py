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

# --- Definitive SSL Certificate Fix ---
# This must be at the very top to configure SSL to use certifi's trusted
# certificates before any network libraries (like requests) are imported.
try:
    ssl._create_default_https_context = ssl._create_unverified_context
    os.environ['SSL_CERT_FILE'] = certifi.where()
except Exception as e:
    logging.error(f"Failed to apply SSL patch: {e}")

# --- Library Imports ---
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


# --- System Path and Local Imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.llm import get_llm_model, PROVIDER_MAP
from models.embeddings import get_huggingface_embeddings
from utils.rag_utils import get_vector_store, format_docs_with_sources
from config.config import TAVILY_API_KEY

# --- Basic Configuration ---
st.set_page_config(page_title="Basic Chatbot", page_icon="🤖", layout="wide")
logging.basicConfig(filename='error.log', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Caching ---
# Cache the embeddings model across the entire session.
@st.cache_resource
def get_embeddings_model_cached():
    """Returns a cached instance of the HuggingFace embeddings model."""
    return get_huggingface_embeddings()

# Cache the vector store based on file content. Reruns only if the file changes.
@st.cache_data(max_entries=5, ttl=3600)
def create_vector_store_cached(_file_hash, chunk_size, chunk_overlap):
    """Creates and caches a vector store from the uploaded document."""
    if "uploaded_file_path" in st.session_state:
        try:
            # Use the cached embeddings model
            embeddings = get_embeddings_model_cached()
            return get_vector_store(st.session_state.uploaded_file_path, chunk_size, chunk_overlap, embeddings)
        except Exception as e:
            st.error(f"Failed to process document: {e}")
            logging.error(f"Vector store creation failed for hash {_file_hash}: {e}")
    return None

# --- Agent and Tools ---
def create_agent(chat_model, vector_store, response_mode):
    """Creates and configures the ReAct agent and its tools."""
    if not TAVILY_API_KEY:
        raise ValueError("Tavily API key is missing. Please set it in your .env file.")

    # 1. Web Search Tool
    search_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)

    # 2. Document Search Tool (Retriever)
    retriever = vector_store.as_retriever()
    doc_tool = Tool(
        name="document_search",
        func=retriever.invoke,
        description="Searches ONLY the content of the uploaded document. Input should be a concise search query."
    )

    tools = [search_tool, doc_tool]

    # Pull the base ReAct prompt from LangChain Hub
    prompt = hub.pull("hwchase17/react")

    # Proactively add response style instructions to the prompt
    mode_instructions = (
        "Respond in a detailed, multi-paragraph format."
        if response_mode == "Detailed"
        else "Respond in a concise, direct format, ideally in 3 sentences or less."
    )
    # Safely insert instructions into the prompt template
    prompt.template = f"RESPONSE STYLE: {mode_instructions}\n\n" + prompt.template

    agent = create_react_agent(chat_model, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        return_intermediate_steps=True,
        handle_parsing_errors="Check your output and make sure it conforms to the correct format!"
    )

# --- UI Rendering ---
def render_sidebar():
    """Renders the sidebar UI elements and returns their current state."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # LLM Provider and Model Selection
        provider_options = [f"{p} {'✅' if details['key'] else '❌'}" for p, details in PROVIDER_MAP.items()]
        selected_option = st.selectbox("LLM Provider", provider_options)

        # FIX: Correctly parse the provider name from the selected option
        provider = ""
        for p_key in PROVIDER_MAP.keys():
            if selected_option.startswith(p_key):
                provider = p_key
                break

        model_name = st.selectbox("Model", PROVIDER_MAP[provider].get("models", []))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05, help="Controls randomness. Lower is more deterministic.")

        st.divider()
        st.header("📄 Document")
        uploaded_file = st.file_uploader("Upload a .pdf, .docx, or .txt file", type=["pdf", "docx", "txt"], label_visibility="collapsed")

        with st.expander("RAG Configuration"):
            chunk_size = st.slider("Chunk Size", 500, 2000, 1000, help="The size of text chunks for the vector store.")
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, help="How much text overlaps between chunks.")

        st.divider()
        st.header("Interface Settings")
        response_mode = st.radio("Response Style", ["Concise", "Detailed"], horizontal=True)
        st.session_state.tts_enabled = st.toggle("Enable Voice Reader 📢", value=False)

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
            st.session_state.messages = []
            # Clear relevant session state to allow for new file processing
            keys_to_clear = ["vector_store", "uploaded_file_path", "uploaded_file_hash"]
            for key in keys_to_clear:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

        return uploaded_file, chunk_size, chunk_overlap, provider, model_name, temperature, response_mode

# --- Text-to-Speech ---
def text_to_speech(text):
    """Converts text to speech and returns the audio bytes."""
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        logging.error(f"TTS failed: {e}")
        st.warning("Could not generate audio response.")
    return None

# --- Main App Logic ---
def main():
    st.title("🤖 Basic Chatbot") # FIX: Renamed the chatbot
    st.session_state.setdefault("messages", [])

    # Render sidebar and get settings
    uploaded_file, chunk_size, chunk_overlap, provider, model_name, temperature, response_mode = render_sidebar()

    # Process uploaded file if it's new
    if uploaded_file:
        file_content = uploaded_file.getvalue()
        file_hash = hashlib.md5(file_content).hexdigest()
        # Check if this is a new file
        if st.session_state.get("uploaded_file_hash") != file_hash:
            with st.spinner("📄 Processing document... Please wait."):
                temp_dir = "temp_files"
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(file_content)

                st.session_state.uploaded_file_path = temp_path
                st.session_state.uploaded_file_hash = file_hash
                # Trigger the cached function to run and store the result
                st.session_state.vector_store = create_vector_store_cached(file_hash, chunk_size, chunk_overlap)
            st.success("Document processed successfully!")

    # Display initial message or chat history
    if not st.session_state.messages:
        st.info("Welcome! Ask a general question or upload a document to query its content.")

    for msg in st.session_state.messages:
        role = "user" if isinstance(msg, HumanMessage) else "assistant"
        with st.chat_message(role):
            st.markdown(msg.content)

    # Handle new user input
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").markdown(prompt)

        with st.chat_message("assistant"):
            if not st.session_state.get("vector_store"):
                st.warning("Please upload a document to use the document search feature.")
                # Allow general conversation even without a file

            # Centralized block for robust error handling
            output = ""
            try:
                with st.spinner("🧠 Thinking..."):
                    # Get model; this will fail gracefully if the key is missing
                    chat_model = get_llm_model(provider, model_name, temperature=temperature)

                    # Use a dummy retriever if no document is uploaded
                    vector_store = st.session_state.get("vector_store")
                    if not vector_store:
                        from langchain_core.retrievers import BaseRetriever
                        from langchain_core.documents import Document
                        class DummyRetriever(BaseRetriever):
                            def _get_relevant_documents(self, query): return []
                        vector_store = type('obj', (object,), {'as_retriever': DummyRetriever})()

                    agent_executor = create_agent(chat_model, vector_store, response_mode)

                    # Invoke agent
                    response = agent_executor.invoke({
                        "input": prompt,
                        "chat_history": st.session_state.messages[-10:]
                    })
                    output = response.get("output", "I encountered an issue. Please try again.")
                    st.markdown(output)

                    # Generate and play audio if enabled
                    if st.session_state.get("tts_enabled"):
                        audio_bytes = text_to_speech(output)
                        if audio_bytes:
                            st.audio(audio_bytes, format="audio/mp3")

                    # Display sources if available
                    if response.get("intermediate_steps"):
                        with st.expander("🔍 View Sources", expanded=False):
                            for i, (agent_action, result) in enumerate(response["intermediate_steps"]):
                                # FIX: Removed the unsupported 'key' argument from st.info
                                st.info(f"**Tool Used:** `{agent_action.tool}`")
                                if agent_action.tool == "document_search":
                                    st.text_area("Retrieved Document Content:", value=format_docs_with_sources(result), height=200, disabled=True, key=f"doc_source_{i}")
                                else: # Web search result
                                    st.json(result, key=f"web_source_{i}")
                
                # Append successful response to history
                st.session_state.messages.append(AIMessage(content=output))

            # FIX: Specific handling for API quota/rate limit errors
            except (GroqRateLimitError, OpenAIRateLimitError, ResourceExhausted) as e:
                error_message = f"⚠️ **API Quota Exceeded:** The `{provider}` API has reached its rate limit. Please check your plan, wait a moment, or try another provider."
                logging.error(f"Quota Error for {provider}: {e}")
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))
            except Exception as e:
                error_message = f"An unexpected error occurred: {e}"
                logging.error(f"Unexpected error in main loop: {e}", exc_info=True)
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))

if __name__ == "__main__":
    main()

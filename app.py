# app.py

# --- Streamlit Page Config: MUST BE FIRST Streamlit command ---
import streamlit as st
st.set_page_config(page_title="AI Mentor Chatbot", page_icon="🤖", layout="wide")


# --- Standard & External Library Imports ---
import os
import sys
import hashlib
from io import BytesIO
import ssl
import certifi
import nltk
import tempfile
import logging

from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from gtts import gTTS
from openai import RateLimitError as OpenAIRateLimitError
from google.api_core.exceptions import ResourceExhausted
from groq import RateLimitError as GroqRateLimitError


# --- SSL/NLTK Setup for Cloud Deployment ---
# This block is critical for ensuring the app works on Streamlit Cloud.
def configure_deployment_environment():
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ["SSL_CERT_FILE"] = certifi.where()
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

configure_deployment_environment()


# --- Local Project Imports ---
# This ensures the app can find the local modules.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.llm import get_llm, PROVIDER_MAP
from models.embeddings import get_embedding_model
from utils.rag_utils import create_vector_store_from_upload
from config.config import TAVILY_API_KEY


# --- Session State Initialization ---
def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "uploaded_file_hash" not in st.session_state:
        st.session_state.uploaded_file_hash = None

initialize_session_state()


# --- Caching ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Loads and caches the embedding model."""
    return get_embedding_model()


# --- Agent & Tool Creation ---
def create_agent_executor(llm, response_style, vector_store=None):
    """Creates the LangChain agent and executor."""
    tools = []
    if TAVILY_API_KEY:
        tavily_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
        tavily_tool.name = "search"
        tavily_tool.description = "A search engine for current events or real-time information."
        tools.append(tavily_tool)
    
    if vector_store:
        retriever = vector_store.as_retriever()
        tools.append(Tool(name="document_search", func=retriever.invoke, description="Searches information from the uploaded document."))

    prompt = hub.pull("hwchase17/react")
    style_prompt = "Respond in a detailed, multi-paragraph format." if response_style == "Detailed" else "Respond concisely, in 3 sentences or less."
    prompt.template = f"RESPONSE STYLE: {style_prompt}\n\n{prompt.template}"
    
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True, return_intermediate_steps=True)


# --- UI Rendering ---
def render_sidebar():
    """Renders the sidebar UI and returns user settings."""
    with st.sidebar:
        st.header("⚙️ Settings")
        available_providers = [p for p, d in PROVIDER_MAP.items() if d["key"]]
        if not available_providers:
            st.error("No LLM API keys found! Add them to your environment secrets.")
            st.stop()
        
        provider = st.selectbox("LLM Provider", available_providers)
        model_name = st.selectbox("Model", PROVIDER_MAP[provider]["models"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        st.divider()
        st.header("📄 RAG Document")
        uploaded_file = st.file_uploader("Upload (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
        st.divider()
        st.header("Interface")
        response_style = st.radio("Response Style", ["Concise", "Detailed"], horizontal=True)
        tts_enabled = st.toggle("Enable Voice Reader 📢", value=False)
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.clear(); st.rerun()

    return provider, model_name, temperature, response_style, tts_enabled, uploaded_file


# --- Core Logic Functions ---
def handle_document_upload(uploaded_file):
    """Processes an uploaded file and updates the vector store in session state."""
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.uploaded_file_hash != file_hash:
            with st.spinner("📄 Processing document..."):
                # Use a secure temporary file
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    embeddings = load_embedding_model()
                    st.session_state.vector_store = create_vector_store_from_upload(tmp_path, embeddings)
                    st.session_state.uploaded_file_hash = file_hash
                    st.sidebar.success("Document processed successfully!")
                except Exception as e:
                    st.sidebar.error(f"Error: {e}")
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path) # Clean up the temp file

def handle_chat_interaction(prompt, llm, response_style, tts_enabled):
    """Handles the user's chat prompt, invokes the agent, and displays the response."""
    st.session_state.messages.append(HumanMessage(content=prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            agent_executor = create_agent_executor(llm, response_style, st.session_state.vector_store)
            with st.spinner("🧠 Thinking..."):
                response = agent_executor.invoke({"input": prompt, "chat_history": st.session_state.messages})
            
            output = response.get("output", "I encountered an error.")
            st.markdown(output)

            if tts_enabled:
                try:
                    tts = gTTS(text=output, lang='en')
                    fp = BytesIO()
                    tts.write_to_fp(fp)
                    st.audio(fp.getvalue(), format="audio/mp3")
                except Exception as e:
                    st.error(f"Text-to-speech failed: {e}")

            if response.get("intermediate_steps"):
                with st.expander("🔍 View Sources"):
                    st.json(response["intermediate_steps"])
            
            st.session_state.messages.append(AIMessage(content=output))

        except (OpenAIRateLimitError, ResourceExhausted, GroqRateLimitError) as e:
            error_message = f"**API Quota Exceeded:** Please check your billing details or try another provider. Error: {e}"
            st.error(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))


# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title("🤖 AI Mentor Chatbot")

    provider, model_name, temperature, response_style, tts_enabled, uploaded_file = render_sidebar()
    
    handle_document_upload(uploaded_file)

    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

    if prompt := st.chat_input("Ask your question here..."):
        try:
            llm = get_llm(provider, model_name, temperature)
            handle_chat_interaction(prompt, llm, response_style, tts_enabled)
        except Exception as e:
            st.error(f"Failed to initialize the language model: {e}")

if __name__ == "__main__":
    main()

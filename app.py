# app.py

# --- Streamlit Page Config: MUST BE FIRST Streamlit command ---
import streamlit as st
st.set_page_config(page_title="Basic Bot", page_icon="🤖", layout="wide")


# --- Hot-patch for sqlite3 on Streamlit Cloud ---
__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')


# --- Standard & External Library Imports ---
import os
import hashlib
from io import BytesIO
import ssl
import certifi
import nltk
import tempfile
import time
from typing import Generator, List, Dict, Any
from queue import Queue

from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from langchain.callbacks.base import BaseCallbackHandler
from gtts import gTTS
from openai import RateLimitError as OpenAIRateLimitError
from google.api_core.exceptions import ResourceExhausted
from groq import RateLimitError as GroqRateLimitError


# --- SSL/NLTK Setup for Cloud Deployment ---
def configure_deployment_environment():
    try:
        ssl._create_default_https_context = ssl._create_unverified_context
        os.environ["SSL_CERT_FILE"] = certifi.where()
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt', quiet=True)

configure_deployment_environment()


# --- Local Project Imports ---
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))
from models.llm import get_llm, PROVIDER_MAP
from models.embeddings import get_embedding_model
from utils.rag_utils import create_vector_store_from_upload
from config.config import TAVILY_API_KEY


# --- Custom Callback Handler for Live Logging ---
class StreamlitCallbackHandler(BaseCallbackHandler):
    """A custom callback handler to stream the agent's thoughts to the UI."""
    def __init__(self):
        self.log_queue = Queue()

    def on_agent_action(self, action, **kwargs):
        # Push the agent's 'thought' process to the queue
        self.log_queue.put(f"🤔 **Thought:**\n{action.log.strip()}")

    def on_tool_end(self, output, **kwargs):
        # Push the observation from the tool to the queue
        self.log_queue.put(f"✅ **Observation:**\n{output}")


# --- Session State Initialization ---
def initialize_session_state():
    defaults = {
        "messages": [],
        "vector_store": None,
        "uploaded_file_hash": None,
        "is_generating": False,
        "interrupt_generation": False,
        "memory": ConversationBufferWindowMemory(
            k=5, return_messages=True, memory_key="chat_history", output_key="output"
        )
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)

initialize_session_state()


# --- Caching ---
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    return get_embedding_model()

@st.cache_resource(max_entries=5, ttl=3600, show_spinner="Creating vector store...")
def create_vector_store_cached(_file_hash, chunk_size, chunk_overlap, uploaded_file_path):
    """Creates and caches the vector store based on file content and RAG parameters."""
    try:
        embeddings = load_embedding_model()
        return create_vector_store_from_upload(uploaded_file_path, chunk_size, chunk_overlap, embeddings)
    except Exception as e:
        st.error(f"Failed to create vector store: {e}")
        return None

# --- Agent & Tool Creation ---
def create_agent_executor(llm, response_style, memory, vector_store=None, web_search_only=False):
    """Creates a more robust LangChain agent and executor with dynamic prompts."""
    tools = []
    
    has_document_tool = not web_search_only and vector_store
    
    if has_document_tool:
        retriever = vector_store.as_retriever()
        tools.append(Tool(
            name="document_search",
            func=retriever.invoke,
            description="Searches the user's uploaded private document. This is the primary tool for answering questions."
        ))
    
    if TAVILY_API_KEY:
        tavily_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
        tavily_tool.name = "search"
        tavily_tool.description = "A general web search engine. Use this ONLY if the document_search tool fails to find a relevant answer."
        tools.append(tavily_tool)

    prompt = hub.pull("hwchase17/react")
    
    if has_document_tool:
        tool_instructions = "You MUST prioritize using the `document_search` tool for questions about the uploaded document. Only use the web 'search' tool if the answer is not found in the document."
    else:
        tool_instructions = "You only have access to a web 'search' tool."

    prompt.template = prompt.template.replace(
        "Think about what to do.",
        f"Think about what to do. {tool_instructions}"
    )
    
    style = "Respond in a detailed, multi-paragraph format." if response_style == "Detailed" else "Respond concisely, in 3 sentences or less."
    prompt = prompt.partial(response_style=style)

    agent = create_react_agent(llm, tools, prompt)
    
    return AgentExecutor(
        agent=agent,
        tools=tools,
        memory=memory,
        verbose=True,
        handle_parsing_errors="I had trouble understanding that. Could you please rephrase?",
        return_intermediate_steps=True,
        max_iterations=15,
        early_stopping_method="generate"
    )


# --- UI Rendering ---
def render_sidebar():
    """Renders the sidebar UI and returns user settings."""
    with st.sidebar:
        st.header("⚙️ Settings")
        
        available_providers = [p for p, d in PROVIDER_MAP.items() if d["key"]]
        if not available_providers:
            st.error("No LLM API keys found!")
            st.stop()
        
        provider = st.selectbox("LLM Provider", available_providers)
        model_name = st.selectbox("Model", PROVIDER_MAP.get(provider, {}).get("models", []))
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)
        
        st.divider()
        st.header("📄 Document & Search")
        uploaded_file = st.file_uploader("Upload a document (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])
        
        with st.expander("RAG Configuration"):
            chunk_size = st.slider("Chunk Size", 500, 4000, 1000, 100)
            chunk_overlap = st.slider("Chunk Overlap", 0, 500, 200, 50)

        web_search_only = st.toggle(
            "Web Search Only", value=False,
            help="If enabled, the bot will only use web search and ignore the document.",
            disabled=not TAVILY_API_KEY
        )
        if not TAVILY_API_KEY:
            st.warning("Tavily API key not found. Web search is disabled.", icon="⚠️")
        
        st.divider()
        st.header("Interface")
        response_style = st.radio("Response Style", ["Concise", "Detailed"], horizontal=True)
        tts_enabled = st.toggle("Enable Voice Reader 📢", value=False)
        
        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True, type="primary"):
            st.session_state.clear()
            st.rerun()

    return provider, model_name, temperature, response_style, tts_enabled, uploaded_file, web_search_only, chunk_size, chunk_overlap


# --- Core Logic Functions ---
def handle_document_upload(uploaded_file, chunk_size, chunk_overlap):
    """Processes an uploaded file and updates the vector store in session state."""
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.uploaded_file_hash != file_hash:
            start_time = time.time()
            with st.spinner("📄 Processing document..."):
                with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[-1]) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                try:
                    st.session_state.vector_store = create_vector_store_cached(
                        file_hash, chunk_size, chunk_overlap, tmp_path
                    )
                    st.session_state.uploaded_file_hash = file_hash
                    processing_time = time.time() - start_time
                    st.sidebar.success(f"Document processed in {processing_time:.2f} seconds.")
                except Exception as e:
                    st.sidebar.error(f"Error processing document: {e}")
                    st.session_state.vector_store = None
                    st.session_state.uploaded_file_hash = None
                finally:
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

def display_final_sources(intermediate_steps):
    """Displays the final, cleaned-up sources after generation is complete."""
    with st.expander("🔍 View Sources"):
        if not intermediate_steps:
            st.info("No sources were used for this response.")
            return

        for i, step in enumerate(intermediate_steps):
            action, observation = step
            with st.container(border=True):
                st.markdown(f"**Step {i+1}: Using tool `{action.tool}`**")
                try:
                    tool_input = action.log.split('Action Input:')[1].strip()
                    st.markdown(f"**Tool Input:**")
                    st.code(tool_input, language='text')
                except IndexError:
                    st.markdown("**Tool Input:** `(Not available)`")
                
                st.markdown("**Observation:**")
                st.info(str(observation))


def stream_agent_response(agent_executor, prompt, callback_handler) -> Generator:
    """Streams the agent's response, yielding output chunks and live logs."""
    full_response = ""
    intermediate_steps = []

    for chunk in agent_executor.stream(
        {"input": prompt},
        config={"callbacks": [callback_handler]}
    ):
        if st.session_state.get("interrupt_generation", False):
            break

        if "output" in chunk:
            output_chunk = chunk["output"]
            full_response += output_chunk
            yield {"output": output_chunk}
        
        while not callback_handler.log_queue.empty():
            yield {"log": callback_handler.log_queue.get()}
            
        if "intermediate_steps" in chunk:
            intermediate_steps = chunk["intermediate_steps"]
    
    yield {"full_response": full_response, "intermediate_steps": intermediate_steps}


def handle_chat_interaction(prompt, llm, response_style, tts_enabled, web_search_only):
    """Handles the user's chat prompt, invokes the agent, and displays the response."""
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.session_state.is_generating = True
    st.session_state.interrupt_generation = False
    
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        start_time = time.time()
        
        response_placeholder = st.empty()
        log_expander = st.expander("🤖 Live Thought Process...")
        log_placeholder = log_expander.empty()
        
        st.button("Stop Generation", key="stop_button", on_click=lambda: st.session_state.update(interrupt_generation=True))

        try:
            callback_handler = StreamlitCallbackHandler()
            agent_executor = create_agent_executor(
                llm, response_style, st.session_state.memory, st.session_state.vector_store, web_search_only
            )
            
            stream_generator = stream_agent_response(agent_executor, prompt, callback_handler)
            
            full_response, log_messages, final_data = "", [], {}
            for chunk in stream_generator:
                if "output" in chunk:
                    full_response += chunk["output"]
                    response_placeholder.markdown(full_response + "▌")
                elif "log" in chunk:
                    log_messages.append(chunk["log"])
                    log_placeholder.info("\n\n---\n\n".join(log_messages))
                else: # Final chunk
                    final_data = chunk

            response_placeholder.markdown(full_response)
            st.session_state.memory.save_context({"input": prompt}, {"output": full_response})
            st.session_state.messages.append(AIMessage(content=full_response))
            
            response_time = time.time() - start_time
            st.caption(f"Response generated in {response_time:.2f} seconds.")

            display_final_sources(final_data.get("intermediate_steps", []))

            if tts_enabled and full_response:
                try:
                    tts = gTTS(text=full_response, lang='en')
                    fp = BytesIO()
                    tts.write_to_fp(fp)
                    st.audio(fp.getvalue(), format="audio/mp3")
                except Exception as e:
                    st.error(f"TTS failed: {e}")

        except (OpenAIRateLimitError, ResourceExhausted, GroqRateLimitError) as e:
            error_message = f"**API Quota Exceeded:** Please check your billing details. Error: {e}"
            st.error(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))
        except Exception as e:
            error_message = f"An unexpected error occurred: {e}"
            st.error(error_message)
            st.session_state.messages.append(AIMessage(content=error_message))
        finally:
            st.session_state.is_generating = False
            st.rerun()


# --- Main Application ---
def main():
    """Main function to run the Streamlit app."""
    st.title("🤖 Basic Bot")

    provider, model_name, temperature, response_style, tts_enabled, uploaded_file, web_search_only, chunk_size, chunk_overlap = render_sidebar()
    
    handle_document_upload(uploaded_file, chunk_size, chunk_overlap)

    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

    if st.session_state.is_generating:
        st.chat_input("Ask your question here...", disabled=True)
    elif prompt := st.chat_input("Ask your question here..."):
        try:
            llm = get_llm(provider, model_name, temperature)
            handle_chat_interaction(prompt, llm, response_style, tts_enabled, web_search_only)
        except Exception as e:
            st.error(f"Failed to initialize the language model: {e}")

if __name__ == "__main__":
    main()

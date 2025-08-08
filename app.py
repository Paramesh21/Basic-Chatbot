# app.py

import streamlit as st
import hashlib
from io import BytesIO

# --- Core LangChain/LLM Imports ---
from langchain_core.messages import AIMessage, HumanMessage
from langchain.agents import create_react_agent, AgentExecutor
from langchain.tools import Tool
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain import hub
from gtts import gTTS

# --- Local Project Imports ---
from config.config import TAVILY_API_KEY
from models.llm import PROVIDER_MAP, get_llm
from models.embeddings import get_embedding_model
from utils.rag_utils import create_vector_store_from_upload

# --- Streamlit Page Configuration ---
st.set_page_config(page_title="AI Mentor Chatbot", page_icon="🤖", layout="wide")


# --- Caching Functions ---

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Loads the embedding model and caches it."""
    return get_embedding_model()

def process_document(_file, embeddings):
    """Processes the uploaded document and creates a vector store."""
    return create_vector_store_from_upload(_file, embeddings)


# --- Agent & Tool Creation ---

def create_rag_retriever(vector_store):
    """Creates a retriever tool from the vector store."""
    retriever = vector_store.as_retriever()
    return Tool(
        name="document_search",
        func=retriever.invoke,
        description="Searches and returns relevant information from the uploaded document.",
    )

def create_agent_executor(llm, response_style, vector_store=None):
    """Creates the LangChain agent and executor."""
    tools = []
    
    # 1. Web Search Tool (Tavily)
    if TAVILY_API_KEY:
        tavily_tool = TavilySearchResults(max_results=3, api_key=TAVILY_API_KEY)
        # Rename the tool to 'search' which the ReAct agent prompt expects.
        tavily_tool.name = "search"
        # Provide a clear description for the agent.
        tavily_tool.description = "A search engine useful for when you need to answer questions about current events or real-time information."
        tools.append(tavily_tool)
    else:
        st.warning("Tavily API key not found. Web search is disabled.", icon="⚠️")

    # 2. Document Search Tool (RAG)
    if vector_store:
        tools.append(create_rag_retriever(vector_store))

    # 3. Agent Prompt
    prompt = hub.pull("hwchase17/react")
    style_prompt = (
        "Respond in a detailed, multi-paragraph format."
        if response_style == "Detailed"
        else "Respond concisely, in 3 sentences or less."
    )
    prompt.template = f"RESPONSE STYLE: {style_prompt}\n\n{prompt.template}"

    # 4. Create Agent and Executor
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        return_intermediate_steps=True,
    )


# --- UI & Main Application Logic ---

def render_sidebar():
    """Renders the sidebar UI elements and returns the settings."""
    with st.sidebar:
        st.header("⚙️ Settings")

        # LLM Provider and Model Selection
        available_providers = [p for p, d in PROVIDER_MAP.items() if d["key"]]
        if not available_providers:
            st.error("No LLM API keys found! Please add them in your environment.")
            st.stop()
        
        provider = st.selectbox("LLM Provider", available_providers)
        model_name = st.selectbox("Model", PROVIDER_MAP[provider]["models"])
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7, 0.05)

        st.divider()

        # RAG Document Upload
        st.header("📄 RAG Document")
        uploaded_file = st.file_uploader("Upload (.pdf, .docx, .txt)", type=["pdf", "docx", "txt"])

        st.divider()

        # Interface Settings
        st.header("Interface")
        response_style = st.radio("Response Style", ["Concise", "Detailed"], horizontal=True)
        tts_enabled = st.toggle("Enable Voice Reader 📢", value=False)

        st.divider()
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.clear()
            st.rerun()

    return provider, model_name, temperature, response_style, tts_enabled, uploaded_file


def text_to_speech(text):
    """Generates audio from text and returns it as bytes."""
    try:
        tts = gTTS(text=text, lang='en')
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return fp.read()
    except Exception as e:
        st.error(f"Text-to-speech failed: {e}")
        return None


def main():
    """Main function to run the Streamlit app."""
    st.title("🤖 AI Mentor Chatbot")

    # --- Initialize Session State ---
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "uploaded_file_hash" not in st.session_state:
        st.session_state.uploaded_file_hash = None


    # --- Sidebar and Settings ---
    provider, model_name, temperature, response_style, tts_enabled, uploaded_file = render_sidebar()

    # --- RAG Processing ---
    if uploaded_file:
        file_hash = hashlib.md5(uploaded_file.getvalue()).hexdigest()
        if st.session_state.uploaded_file_hash != file_hash:
            embeddings = load_embedding_model()
            st.session_state.vector_store = process_document(uploaded_file, embeddings)
            st.session_state.uploaded_file_hash = file_hash
            if st.session_state.vector_store:
                st.sidebar.success("Document processed successfully!")
            else:
                st.sidebar.error("Failed to process document.")

    # --- Chat History Display ---
    for message in st.session_state.messages:
        role = "assistant" if isinstance(message, AIMessage) else "user"
        with st.chat_message(role):
            st.markdown(message.content)

    # --- User Input and Agent Execution ---
    if prompt := st.chat_input("Ask your question here..."):
        st.session_state.messages.append(HumanMessage(content=prompt))
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            try:
                llm = get_llm(provider, model_name, temperature)
                agent_executor = create_agent_executor(llm, response_style, st.session_state.vector_store)
                
                with st.spinner("🧠 Thinking..."):
                    response = agent_executor.invoke(
                        {"input": prompt, "chat_history": st.session_state.messages}
                    )
                
                output = response.get("output", "I encountered an error.")
                st.markdown(output)
                st.session_state.messages.append(AIMessage(content=output))

                if tts_enabled:
                    audio_bytes = text_to_speech(output)
                    if audio_bytes:
                        st.audio(audio_bytes, format="audio/mp3")

                if response.get("intermediate_steps"):
                    with st.expander("🔍 View Sources"):
                        st.json(response["intermediate_steps"])

            except Exception as e:
                error_message = f"An error occurred: {e}"
                st.error(error_message)
                st.session_state.messages.append(AIMessage(content=error_message))


if __name__ == "__main__":
    main()

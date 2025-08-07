# Basic Chatbot

A multi-functional chatbot built with Streamlit and LangChain. It can answer questions based on uploaded documents (RAG), search the web for real-time information, and switch between multiple LLM providers.

## 🌟 Features

- **Robust RAG Pipeline**: Chat with your documents (`.pdf`, `.docx`, `.txt`).
- **Web Search**: Integrated with Tavily for real-time information.
- **Multi-LLM Support**: Switch between Groq, OpenAI, and Google Gemini models.
- **Configurable UI**: Adjust RAG parameters, response modes, and chat history display length.
- **Source Attribution**: Responses include sources for verification.
- **Performance Caching**: Intelligent caching based on file content.
- **Text-to-Speech**: A global toggle to read responses aloud.

## 🛠️ Setup

### Prerequisites
- Python 3.8+
- For processing some document types, `unstructured` may require system-level dependencies. For details, see the [Unstructured documentation](https://unstructured-io.github.io/unstructured/installing.html).

### Instructions
1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd AI_UseCase
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Configure API Keys:**
    - Rename the `.env.example` file to `.env`.
    - Open the new `.env` file and replace the placeholder strings with your actual API keys.

## 🚀 How to Run

1.  Make sure your virtual environment is activated.
2.  Run the Streamlit application from your terminal:
    ```bash
    streamlit run app.py
    ```
3.  The application will open in your web browser.

## 🚢 Deployment to Streamlit Cloud

1.  Push your project to a GitHub repository (ensure your `.env` file is in your `.gitignore`!).
2.  In your Streamlit Cloud account, create a new app and link it to your GitHub repository.
3.  In the "Advanced settings..." section, go to the **Secrets** tab.
4.  Add your API keys one by one, matching the names in the `.env` file (e.g., `GROQ_API_KEY = "gsk_..."`).
5.  Deploy the app.
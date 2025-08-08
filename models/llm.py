# models/llm.py

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY

PROVIDER_MAP = {
    "Groq": {
        "models": ["llama3-8b-8192", "llama3-70b-8192"],
        "key": GROQ_API_KEY,
        "class": ChatGroq,
    },
    "OpenAI": {
        "models": ["gpt-4o", "gpt-3.5-turbo"],
        "key": OPENAI_API_KEY,
        "class": ChatOpenAI,
    },
    "Google Gemini": {
        "models": ["gemini-1.5-pro-latest", "gemini-1.5-flash-latest"],
        "key": GOOGLE_API_KEY,
        "class": ChatGoogleGenerativeAI,
    },
}

def get_llm(provider: str, model_name: str, temperature: float = 0.7):
    """Initializes and returns the specified chat model instance."""
    if provider not in PROVIDER_MAP:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    provider_info = PROVIDER_MAP[provider]
    if not provider_info["key"]:
        raise ValueError(f"API key for {provider} is missing.")
    if model_name not in provider_info["models"]:
        raise ValueError(f"Model '{model_name}' is not supported by '{provider}'.")

    try:
        model_class = provider_info["class"]
        params = {"temperature": temperature}
        
        # FIX: Use 'model' for Gemini and 'model_name' for others
        if provider == "Google Gemini":
            params["model"] = model_name
            params["google_api_key"] = provider_info["key"]
        else:
            params["model_name"] = model_name
            params["api_key"] = provider_info["key"]
            
        return model_class(**params)
        
    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM '{model_name}': {e}") from e

# models/llm.py

from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from config.config import GROQ_API_KEY, OPENAI_API_KEY, GOOGLE_API_KEY

# A mapping of LLM providers to their models, API keys, and classes.
# This makes it easy to add or update providers in the future.
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
    """
    Initializes and returns the specified chat model instance.

    Args:
        provider: The name of the LLM provider (e.g., "OpenAI").
        model_name: The specific model to use (e.g., "gpt-4o").
        temperature: The creativity level for the model's responses.

    Returns:
        An instance of the specified chat model.

    Raises:
        ValueError: If the provider or model is unsupported, or if the API key is missing.
    """
    if provider not in PROVIDER_MAP:
        raise ValueError(f"Unsupported LLM provider: {provider}")

    provider_info = PROVIDER_MAP[provider]

    if not provider_info["key"]:
        raise ValueError(f"API key for {provider} is missing. Please set it in the deployment environment.")

    if model_name not in provider_info["models"]:
        raise ValueError(f"Model '{model_name}' is not supported by '{provider}'.")

    try:
        model_class = provider_info["class"]
        params = {"model_name": model_name, "temperature": temperature}

        # The API key parameter name differs between providers
        if provider in ["Groq", "OpenAI"]:
            params["api_key"] = provider_info["key"]
        elif provider == "Google Gemini":
            params["google_api_key"] = provider_info["key"]

        return model_class(**params)

    except Exception as e:
        raise RuntimeError(f"Failed to initialize LLM '{model_name}' from {provider}: {e}") from e

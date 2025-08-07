# config/config.py
import os
from dotenv import load_dotenv

load_dotenv()

def get_api_key(key_name: str, placeholder: str) -> str:
    """
    Retrieves an API key, prioritizing environment variables.
    Returns an empty string if the key is invalid or not found.
    """
    key = os.getenv(key_name, placeholder)
    if not key or "YOUR_" in key or key.strip() == "":
        return ""
    return key

GROQ_API_KEY = get_api_key("GROQ_API_KEY", "YOUR_GROQ_API_KEY")
TAVILY_API_KEY = get_api_key("TAVILY_API_KEY", "YOUR_TAVILY_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY", "YOUR_GOOGLE_API_KEY")
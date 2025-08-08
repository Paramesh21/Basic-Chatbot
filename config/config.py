# config/config.py

import os
from dotenv import load_dotenv

# Load environment variables from a .env file if it exists
load_dotenv()

def get_api_key(key_name: str) -> str | None:
    """
    Retrieves an API key from environment variables.
    Returns None if the key is not found or is a placeholder.
    """
    key = os.getenv(key_name)
    if not key or "YOUR_" in key or key.strip() == "":
        return None
    return key

# Load all API keys
GROQ_API_KEY = get_api_key("GROQ_API_KEY")
TAVILY_API_KEY = get_api_key("TAVILY_API_KEY")
OPENAI_API_KEY = get_api_key("OPENAI_API_KEY")
GOOGLE_API_KEY = get_api_key("GOOGLE_API_KEY")

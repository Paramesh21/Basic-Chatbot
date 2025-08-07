# download_nltk_data.py
import nltk
import logging

logging.basicConfig(filename='error.log', level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')

print("Downloading NLTK 'punkt' resource...")
try:
    nltk.download('punkt', quiet=True)
    print("Download complete.")
except Exception as e:
    logging.error(f"NLTK download failed: {e}")
    print(f"An error occurred: {e}")
import os
import google.generativeai as genai
import logging
from typing import List

# Configure the Gemini API key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_text_embeddings(texts: List[str]):
    """
    Generates embeddings for a list of text strings using the Gemini API.

    Args:
        texts (List[str]): A list of text strings to embed.

    Returns:
        A list of embedding vectors.
    """
    if not texts:
        return []

    try:
        logging.info(f"Generating embeddings for {len(texts)} documents.")
        # The 'models/embedding-001' is the recommended model for embeddings
        response = genai.embed_content(
            model="models/embedding-001",
            content=texts,
            task_type="retrieval_document"
        )
        return response['embedding']
    except Exception as e:
        logging.error(f"Error generating embeddings: {e}")
        return []

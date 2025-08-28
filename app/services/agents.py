import os
import logging
import requests
import google.generativeai as genai
from app.services.retrieval import retrieve_documents

# Configure the Gemini API for LLM generation
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Get API keys from environment variables
N8N_WEBHOOK_URL = os.getenv("N8N_WEBHOOK_URL")
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")

def generate_summary(query: str):
    """
    Uses a RAG approach to generate a news summary.
    1. Retrieves relevant documents based on the query.
    2. Constructs a prompt with the retrieved context.
    3. Uses Gemini to generate a summary.

    Args:
        query (str): The user's news query.

    Returns:
        The generated news summary text.
    """
    logging.info(f"Generating summary for query: '{query}'")
    
    # Step 1: Retrieve documents
    retrieved_data = retrieve_documents(query)
    if not retrieved_data:
        logging.warning("No documents retrieved for summary generation.")
        return "No relevant information found to generate a summary."

    context = "\n\n".join([doc_content for doc_content, score in retrieved_data])

    # Step 2: Construct the prompt
    # This is the "fine-tuning" through a well-crafted prompt.
    prompt = f"""
    Based on the following context, summarize the news in a concise and neutral tone.
    Do not include any information that is not explicitly present in the context.

    Context:
    {context}

    Summary:
    """

    try:
        # Step 3: Generate the summary using the Gemini LLM
        model = genai.GenerativeModel('gemini-1.5-flash')
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        logging.error(f"Error generating content with Gemini: {e}")
        return "An error occurred while generating the summary."

def send_news_update(recipient: str, summary: str, method: str):
    """
    Sends the news summary to a recipient via email or SMS using n8n.

    Args:
        recipient (str): The email or phone number of the recipient.
        summary (str): The news summary text.
        method (str): The delivery method ('email' or 'sms').
    """
    if not N8N_WEBHOOK_URL:
        logging.error("N8N webhook URL is not configured.")
        return

    payload = {
        "recipient": recipient,
        "summary": summary,
        "method": method
    }

    try:
        logging.info(f"Sending {method} update to {recipient} via n8n webhook.")
        response = requests.post(N8N_WEBHOOK_URL, json=payload, timeout=10)
        response.raise_for_status() # Raise an exception for bad status codes
        logging.info(f"Successfully triggered n8n webhook. Response: {response.text}")
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to send update via n8n: {e}")

def convert_to_audio(summary: str):
    """
    Converts a text summary to an audio file using ElevenLabs.

    Args:
        summary (str): The news summary text.

    Returns:
        The path to the saved audio file, or None if an error occurs.
    """
    if not ELEVENLABS_API_KEY:
        logging.error("ElevenLabs API key is not configured.")
        return None
    
    url = "https://api.elevenlabs.io/v1/text-to-speech/<voice_id>"
    headers = {
        "xi-api-key": ELEVENLABS_API_KEY,
        "Content-Type": "application/json"
    }
    data = {
        "text": summary,
        "model_id": "eleven_multilingual_v2"
    }

    try:
        logging.info("Converting summary to audio using ElevenLabs.")
        response = requests.post(url, json=data, headers=headers, timeout=20, stream=True)
        response.raise_for_status()
        
        # Save the audio stream to a file
        audio_file_path = "summary.mp3"
        with open(audio_file_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=1024):
                f.write(chunk)
        
        logging.info(f"Audio file saved to {audio_file_path}")
        return audio_file_path
    except requests.exceptions.RequestException as e:
        logging.error(f"Failed to convert text to audio with ElevenLabs: {e}")
        return None

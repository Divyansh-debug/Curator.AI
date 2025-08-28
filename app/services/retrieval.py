import os
import logging
import google.generativeai as genai
from typing import List, Tuple
from langchain_community.vectorstores import FAISS

# Configure the Gemini API for query embedding
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_query_embedding(query: str):
    """
    Generates an embedding for a search query using the Gemini API.

    Args:
        query (str): The search query string.

    Returns:
        The embedding vector for the query.
    """
    try:
        logging.info("Generating query embedding.")
        response = genai.embed_content(
            model="models/embedding-001",
            content=query,
            task_type="retrieval_query"
        )
        return response['embedding']
    except Exception as e:
        logging.error(f"Error generating query embedding: {e}")
        return None

def retrieve_documents(query: str, k: int = 5) -> List[Tuple[str, float]]:
    """
    Retrieves the top-k most relevant documents from the FAISS index.

    Args:
        query (str): The user's search query.
        k (int): The number of top documents to retrieve.

    Returns:
        A list of tuples, where each tuple contains the document content
        and its similarity score.
    """
    try:
        if not os.path.exists("faiss_index"):
            logging.error("FAISS index not found. Please ingest documents first.")
            return []

        # Load the FAISS index without an embedding function
        # This is because the embeddings are generated externally by the Gemini API.
        faiss_index = FAISS.load_local("faiss_index", embeddings=None, allow_dangerous_deserialization=True)

        query_embedding = get_query_embedding(query)
        if query_embedding is None:
            return []

        logging.info(f"Retrieving top {k} documents from FAISS index.")
        # Perform a similarity search using the vector directly
        docs_with_scores = faiss_index.similarity_search_with_score_by_vector(query_embedding, k=k)
        
        # Format the output to be more usable
        retrieved_data = [(doc.page_content, score) for doc, score in docs_with_scores]
        return retrieved_data
    except Exception as e:
        logging.error(f"Error retrieving documents: {e}")
        return []

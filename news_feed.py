import os
import logging
import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

# Configure the Gemini API key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()
model = genai.GenerativeModel('gemini-1.5-flash')

# Initialize the embedding model (must be the same as ingestion)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Pydantic model for the news request body
class NewsRequest(BaseModel):
    query: str
    num_results: int = 5

@router.post("/get-news-summary/")
async def get_news_summary(request: NewsRequest):
    """
    Generates a news summary based on a query using RAG and Gemini.
    """
    try:
        # Load the FAISS index
        faiss_index = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        
        # Perform a semantic search on the index
        retrieved_docs = faiss_index.similarity_search(request.query, k=request.num_results)
        
        if not retrieved_docs:
            return {"status": "info", "message": "No relevant documents found in the database."}
            
        # Combine the retrieved document content into a single context string
        context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        
        # Create a prompt for the Gemini model
        prompt = f"""
        Based on the following context, summarize the news in a concise and neutral tone.
        
        Context:
        {context}
        
        Summary:
        """
        
        # Generate the summary using the Gemini API
        response = model.generate_content(prompt)
        
        return {
            "status": "success",
            "query": request.query,
            "summary": response.text.strip(),
            "source_documents": [doc.metadata for doc in retrieved_docs] # Optional: return source info
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="The FAISS database is not available. Please ingest documents first.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred.")

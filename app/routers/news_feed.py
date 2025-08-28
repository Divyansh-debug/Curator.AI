import os
import logging
import google.generativeai as genai
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from langchain_community.vectorstores import FAISS

# Configure the Gemini API key from environment variables
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

router = APIRouter()
model = genai.GenerativeModel('gemini-1.5-flash')

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
        # We use `allow_dangerous_deserialization=True` because the index
        # was not created with a LangChain-compatible embedding function.
        faiss_index = FAISS.load_local("faiss_index", embeddings=None, allow_dangerous_deserialization=True)
        
        # Embed the user's query using the Gemini API
        query_embedding_response = genai.embed_content(
            model="models/embedding-001",
            content=request.query,
            task_type="retrieval_query"
        )
        query_embedding = query_embedding_response['embedding']
        
        # Perform a semantic search on the index
        # We pass the raw query embedding to the FAISS search function
        retrieved_docs_and_scores = faiss_index.similarity_search_with_score_by_vector(query_embedding, k=request.num_results)
        
        if not retrieved_docs_and_scores:
            return {"status": "info", "message": "No relevant documents found in the database."}
            
        # Combine the retrieved document content into a single context string
        context = "\n\n".join([doc.page_content for doc, score in retrieved_docs_and_scores])
        
        # Create a prompt for the Gemini model
        prompt = f"""
        Given the following context from various news sources, summarize the information concisely and neutrally.
        Do not add any information that is not present in the context.
        
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
            # Optional: Return a list of the content of the top documents
            "source_documents_preview": [doc.page_content for doc, score in retrieved_docs_and_scores]
        }
        
    except FileNotFoundError:
        raise HTTPException(status_code=503, detail="The FAISS database is not available. Please ingest documents first.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred.")

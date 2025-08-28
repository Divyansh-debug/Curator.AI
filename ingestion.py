import os
import requests
from bs4 import BeautifulSoup
from fastapi import APIRouter, File, UploadFile, HTTPException
from PyPDF2 import PdfReader
from docx import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)

router = APIRouter()

# Initialize the embedding model
# You can use a local model for cost-effectiveness
# The "all-MiniLM-L6-v2" model is a good general-purpose choice.
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Define a function to process documents and create a FAISS index
def process_and_index_documents(texts: list[str], index_name: str = "faiss_index"):
    """
    Processes a list of text documents, creates embeddings, and stores them in a FAISS index.
    
    Args:
        texts (list[str]): A list of text strings to be processed.
        index_name (str): The name for the FAISS index file.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    docs = text_splitter.create_documents(texts)
    
    if not docs:
        logging.error("No documents were created. Check the input text.")
        return

    logging.info(f"Creating a FAISS index with {len(docs)} documents.")
    db = FAISS.from_documents(docs, embeddings)
    db.save_local(index_name)
    logging.info(f"FAISS index saved to '{index_name}'.")

# File Upload Endpoint
@router.post("/ingest-documents/")
async def ingest_documents(files: list[UploadFile] = File(...)):
    """
    Ingests PDF and Word documents, extracts text, and creates a FAISS index.
    """
    all_text = []
    
    for file in files:
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        try:
            if file_extension == ".pdf":
                reader = PdfReader(file.file)
                text = ""
                for page in reader.pages:
                    text += page.extract_text()
                all_text.append(text)
                
            elif file_extension in [".docx", ".doc"]:
                doc = Document(file.file)
                text = ""
                for paragraph in doc.paragraphs:
                    text += paragraph.text + "\n"
                all_text.append(text)
                
            else:
                logging.warning(f"Skipping unsupported file type: {file.filename}")
                continue
                
        except Exception as e:
            logging.error(f"Error processing file {file.filename}: {e}")
            raise HTTPException(status_code=500, detail=f"Error processing file {file.filename}")
            
    if all_text:
        process_and_index_documents(all_text)
        return {"status": "success", "message": f"Successfully ingested {len(files)} documents."}
    else:
        return {"status": "error", "message": "No valid documents were processed."}

# Web Scraping Endpoint
@router.post("/scrape-website/")
async def scrape_website(url: str):
    """
    Scrapes a given URL, extracts text, and updates the FAISS index.
    """
    if not url.startswith(("http://", "https://")):
        raise HTTPException(status_code=400, detail="URL must start with http:// or https://")
    
    try:
        response = requests.get(url)
        response.raise_for_status() # Raise an exception for bad status codes
        soup = BeautifulSoup(response.text, "html.parser")
        
        # Extract all text from the main body of the page
        # This is a simple approach, can be improved for specific websites
        body_text = soup.body.get_text(separator="\n", strip=True)
        
        # Load the existing index if it exists, otherwise create a new one
        if os.path.exists("faiss_index"):
            logging.info("Loading existing FAISS index...")
            db = FAISS.load_local("faiss_index", embeddings)
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            docs = text_splitter.create_documents([body_text])
            db.add_documents(docs)
            db.save_local("faiss_index")
            logging.info("FAISS index updated.")
        else:
            logging.info("No existing FAISS index found. Creating a new one.")
            process_and_index_documents([body_text])

        return {"status": "success", "message": f"Successfully scraped and ingested data from {url}."}
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Error scraping URL {url}: {e}")
        raise HTTPException(status_code=500, detail=f"Error scraping URL: {e}")

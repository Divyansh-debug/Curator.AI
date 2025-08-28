import logging
from fastapi import FastAPI
from app.routers import ingestion, news_feed

# Set up logging for the main application
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Create the FastAPI app instance
# Add a title, description, and version for the auto-generated OpenAPI docs at /docs
app = FastAPI(
    title="AI News App API",
    description="A Retrieval-Augmented Generation (RAG) API for ingesting, summarizing, and delivering AI news.",
    version="1.0.0"
)

# Include the routers for different parts of the API.
# The 'prefix' and 'tags' help organize the documentation.
app.include_router(ingestion.router, prefix="/ingestion", tags=["Ingestion"])
app.include_router(news_feed.router, prefix="/news", tags=["News Feed"])

@app.get("/")
async def root():
    """
    A simple root endpoint to confirm the API is running.
    """
    logger.info("Root endpoint was accessed.")
    return {"message": "Welcome to the AI News App API! 🚀"}

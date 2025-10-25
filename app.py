from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from api.routes import router
from services import get_search_service, get_embedding_service

# Initialize FastAPI app
app = FastAPI(
    title="Search Service",
    description="Hybrid vector search service for Tro Tot VN",
    version="1.0.0"
)

# CORS middleware (for internal service communication)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Internal network only
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routes
app.include_router(router)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    logger.info("🚀 Starting search-service...")
    
    try:
        # Initialize embedding service (loads model)
        logger.info("Loading embedding model...")
        get_embedding_service()
        
        # Initialize search service (connects to Milvus)
        logger.info("Connecting to Milvus...")
        get_search_service()
        
        logger.info("✅ Search service ready!")
        
    except Exception as e:
        logger.error(f"Failed to initialize search service: {e}")
        raise

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "search-service",
        "status": "running",
        "endpoints": {
            "search": "POST /search/hybrid",
            "health": "GET /health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


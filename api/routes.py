from fastapi import APIRouter, HTTPException
from loguru import logger
from .schemas import SearchRequest, SearchResponse, HealthResponse
from services import get_search_service

router = APIRouter()

@router.post("/search/hybrid", response_model=SearchResponse)
async def hybrid_search(request: SearchRequest):
    """
    Perform hybrid search (dense semantic + sparse BM25)
    
    Returns only post IDs for Node.js to enrich with full data
    """
    try:
        logger.info(f"Search request: query='{request.query}', city={request.city}, limit={request.limit}")
        
        search_service = get_search_service()
        
        results = search_service.hybrid_search(
            query=request.query,
            city=request.city,
            district=request.district,
            ward=request.ward,
            price_min=request.price_min,
            price_max=request.price_max,
            acreage_min=request.acreage_min,
            acreage_max=request.acreage_max,
            interior_condition=request.interior_condition,
            limit=request.limit
        )
        
        return SearchResponse(
            success=True,
            post_ids=results["post_ids"],
            total=results["total"],
            search_time_ms=results["search_time_ms"]
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    try:
        search_service = get_search_service()
        milvus_connected = search_service.collection is not None
        
        return HealthResponse(
            status="ok",
            service="search-service",
            milvus_connected=milvus_connected
        )
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return HealthResponse(
            status="error",
            service="search-service",
            milvus_connected=False
        )


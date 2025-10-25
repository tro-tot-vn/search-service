from pydantic import BaseModel, Field
from typing import Optional, List

class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query text")
    city: Optional[str] = Field(None, description="Filter by city")
    district: Optional[str] = Field(None, description="Filter by district")
    ward: Optional[str] = Field(None, description="Filter by ward")
    price_min: Optional[int] = Field(None, description="Minimum price", ge=0)
    price_max: Optional[int] = Field(None, description="Maximum price", ge=0)
    acreage_min: Optional[int] = Field(None, description="Minimum acreage", ge=0)
    acreage_max: Optional[int] = Field(None, description="Maximum acreage", ge=0)
    interior_condition: Optional[str] = Field(None, description="Interior condition filter")
    limit: int = Field(100, description="Maximum number of results", ge=1, le=500)

class SearchResponse(BaseModel):
    success: bool = True
    post_ids: List[int] = Field(description="List of post IDs matching the search")
    total: int = Field(description="Total number of results")
    search_time_ms: float = Field(description="Search time in milliseconds")

class HealthResponse(BaseModel):
    status: str = "ok"
    service: str = "search-service"
    milvus_connected: bool


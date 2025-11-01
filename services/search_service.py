import time
from typing import List, Optional, Dict, Any
from pymilvus import connections, Collection, AnnSearchRequest, WeightedRanker
from loguru import logger
from config.settings import settings
from .embedding_service import get_embedding_service

class SearchService:
    def __init__(self):
        self.host = settings.MILVUS_HOST
        self.port = settings.MILVUS_PORT
        self.collection_name = "posts_hybrid"
        
        # Connect to Milvus
        connections.connect(
            alias="default",
            host=self.host,
            port=str(self.port)
        )
        logger.info(f"✅ Connected to Milvus at {self.host}:{self.port}")
        
        # Load collection
        self.collection = Collection(self.collection_name)
        self.collection.load()
        logger.info(f"✅ Collection {self.collection_name} loaded")
        
        # Get embedding service
        self.embedding_service = get_embedding_service()
    
    def _build_filter_expression(
        self,
        city: Optional[str] = None,
        district: Optional[str] = None,
        ward: Optional[str] = None,
        price_min: Optional[int] = None,
        price_max: Optional[int] = None,
        acreage_min: Optional[int] = None,
        acreage_max: Optional[int] = None,
        interior_condition: Optional[str] = None
    ) -> str:
        """Build Milvus filter expression from parameters"""
        filters = []
        
        # CRITICAL: Always filter by Approved status
        filters.append('status == "Approved"')
        
        if city:
            filters.append(f'city == "{city}"')
        
        if district:
            filters.append(f'district == "{district}"')
        
        if ward:
            filters.append(f'ward == "{ward}"')
        
        if price_min is not None:
            filters.append(f'price >= {price_min}')
        
        if price_max is not None:
            filters.append(f'price <= {price_max}')
        
        if acreage_min is not None:
            filters.append(f'acreage >= {acreage_min}')
        
        if acreage_max is not None:
            filters.append(f'acreage <= {acreage_max}')
        
        if interior_condition:
            filters.append(f'interior_condition == "{interior_condition}"')
        
        return " && ".join(filters)
    
    def hybrid_search(
        self,
        query: str,
        city: Optional[str] = None,
        district: Optional[str] = None,
        ward: Optional[str] = None,
        price_min: Optional[int] = None,
        price_max: Optional[int] = None,
        acreage_min: Optional[int] = None,
        acreage_max: Optional[int] = None,
        interior_condition: Optional[str] = None,
        limit: int = 100
    ) -> Dict[str, Any]:
        """
        Perform hybrid search (dense + 3 sparse BM25)
        
        Returns:
            {
                "post_ids": [123, 456, 789, ...],
                "total": 85,
                "search_time_ms": 45.2
            }
        """
        start_time = time.time()
        
        # 1. Generate query embedding
        logger.info(f"Generating embedding for query: {query[:50]}...")
        query_embedding = self.embedding_service.generate_dense_embedding(query)
        
        # 2. Build filter expression
        logger.info(f"[BUILD FILTER] city={city}, district={district}, ward={ward}")
        logger.info(f"[BUILD FILTER] price_min={price_min}, price_max={price_max}")
        logger.info(f"[BUILD FILTER] acreage_min={acreage_min}, acreage_max={acreage_max}")
        logger.info(f"[BUILD FILTER] interior_condition={interior_condition}")
        
        filter_expr = self._build_filter_expression(
            city=city,
            district=district,
            ward=ward,
            price_min=price_min,
            price_max=price_max,
            acreage_min=acreage_min,
            acreage_max=acreage_max,
            interior_condition=interior_condition
        )
        
        logger.info(f"[FILTER EXPRESSION] {filter_expr}")
        
        # 3. Create search requests for hybrid search
        # Dense vector search (semantic)
        dense_search_params = {"metric_type": "COSINE", "params": {"ef": 128}}
        
        logger.info(f"DEBUG: query_embedding type: {type(query_embedding)}, len: {len(query_embedding)}")
        logger.info(f"DEBUG: query text: {query}")
        
        dense_req = AnnSearchRequest(
            data=[query_embedding],
            anns_field="dense_vector",
            param=dense_search_params,
            limit=limit,
            expr=filter_expr
        )
        
        # Sparse BM25 searches (keyword matching)
        # For BM25 with built-in Functions, pass query TEXT as data
        sparse_search_params = {"metric_type": "BM25", "params": {}}
        
        logger.info(f"DEBUG: Creating BM25 search request with query: '{query}'")
        
        # Title BM25 - Pass text for BM25 function
        title_req = AnnSearchRequest(
            data=[query],  # Pass text directly for BM25
            anns_field="sparse_title",
            param=sparse_search_params,
            limit=limit,
            expr=filter_expr
        )
        
        # Description BM25
        desc_req = AnnSearchRequest(
            data=[query],  # Pass text directly for BM25
            anns_field="sparse_description",
            param=sparse_search_params,
            limit=limit,
            expr=filter_expr
        )
        
        # Address BM25
        addr_req = AnnSearchRequest(
            data=[query],  # Pass text directly for BM25
            anns_field="sparse_address",
            param=sparse_search_params,
            limit=limit,
            expr=filter_expr
        )
        
        # 4. Perform hybrid search with weighted ranker
        # Weights: 40% dense, 30% title, 20% description, 10% address
        reranker = WeightedRanker(float(0.4), float(0.3), float(0.2), float(0.1))
        
        results = self.collection.hybrid_search(
            reqs=[dense_req, title_req, desc_req, addr_req],
            rerank=reranker,
            limit=limit,
            output_fields=["id"]
        )
        
        # 5. Extract post IDs
        post_ids = []
        if results and len(results) > 0:
            for hit in results[0]:
                post_ids.append(hit.id)
        
        search_time_ms = (time.time() - start_time) * 1000
        
        logger.info(f"✅ Hybrid search completed: {len(post_ids)} results in {search_time_ms:.2f}ms")
        
        return {
            "post_ids": post_ids,
            "total": len(post_ids),
            "search_time_ms": round(search_time_ms, 2)
        }

# Singleton
_search_service = None

def get_search_service():
    global _search_service
    if _search_service is None:
        _search_service = SearchService()
    return _search_service


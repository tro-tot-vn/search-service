# Search Service

Python FastAPI service for hybrid vector search on Milvus.

## Features

- **Hybrid Search**: Combines dense semantic vectors (BGE-M3) with sparse BM25 keyword matching
- **Multi-field BM25**: Separate BM25 scoring for title, description, and address
- **Weighted Ranking**: 40% semantic, 30% title, 20% description, 10% address
- **Scalar Filtering**: Filter by city, district, price, acreage, interior condition
- **Efficient**: Returns only post IDs for Node.js to enrich with full data

## Architecture

```
Node.js Backend → Search Service (HTTP) → Milvus
                                            ↓
                                    Hybrid Search
                              (1 dense + 3 sparse)
```

## API Endpoints

### POST /search/hybrid

Perform hybrid vector search.

**Request:**
```json
{
  "query": "phòng trọ gần ĐH Bách Khoa",
  "city": "Hà Nội",
  "district": "Hai Bà Trưng",
  "price_min": 1000000,
  "price_max": 3000000,
  "acreage_min": 20,
  "acreage_max": 40,
  "interior_condition": "Nội thất đầy đủ",
  "limit": 100
}
```

**Response:**
```json
{
  "success": true,
  "post_ids": [123, 456, 789, ...],
  "total": 85,
  "search_time_ms": 45.2
}
```

### GET /health

Health check endpoint.

**Response:**
```json
{
  "status": "ok",
  "service": "search-service",
  "milvus_connected": true
}
```

## Setup

### Local Development

1. Install dependencies:
```bash
cd search-service
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. Configure environment:
```bash
cp .env.example .env
# Edit .env with your settings
```

3. Run:
```bash
python app.py
# or
uvicorn app:app --reload --host 0.0.0.0 --port 8000
```

### Docker

```bash
# Build
docker build -t search-service .

# Run
docker run -p 8000:8000 \
  -e MILVUS_HOST=localhost \
  -e MILVUS_PORT=19530 \
  search-service
```

### Docker Compose

Already configured in `infras/compose/docker-compose.dev.yaml`:

```bash
cd infras/compose
docker compose -f docker-compose.dev.yaml up search-service
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `MILVUS_HOST` | localhost | Milvus server host |
| `MILVUS_PORT` | 19530 | Milvus server port |
| `REDIS_HOST` | localhost | Redis host (for future caching) |
| `REDIS_PORT` | 5555 | Redis port |
| `EMBEDDING_MODEL` | BAAI/bge-m3 | BGE-M3 model name |
| `EMBEDDING_DIM` | 128 | Embedding dimension |
| `DEVICE` | cpu | Device (cpu/cuda) |
| `DEFAULT_SEARCH_LIMIT` | 100 | Default search limit |
| `MAX_SEARCH_LIMIT` | 500 | Maximum search limit |

## Milvus Collection Schema

The service searches the `posts_hybrid` collection with:

**Vector Fields:**
- `dense_vector`: 128-dim semantic embedding (COSINE similarity)
- `sparse_title`: BM25 sparse vector for title
- `sparse_description`: BM25 sparse vector for description
- `sparse_address`: BM25 sparse vector for address

**Scalar Fields (for filtering):**
- `city`, `district`, `ward`, `street`
- `price`, `acreage`
- `interior_condition`
- `owner_id`, `created_at`, `extended_at`

## Hybrid Search Strategy

1. **Generate Query Embedding**: Use BGE-M3 to create 128-dim vector
2. **Build Filter Expression**: Construct Milvus filter from parameters
3. **Execute Hybrid Search**: 
   - Dense vector search (semantic similarity)
   - 3x sparse BM25 searches (keyword matching)
   - Weighted ranking: 0.4 + 0.3 + 0.2 + 0.1 = 1.0
4. **Return IDs**: Only post IDs, not full data

## Performance

- **Embedding Generation**: ~100-200ms (CPU) / ~10-20ms (GPU)
- **Milvus Search**: ~20-50ms
- **Total**: ~150-250ms (CPU) / ~30-70ms (GPU)

## Integration with Node.js

The Node.js backend calls this service:

```typescript
// Node.js
const result = await axios.post('http://search-service:8000/search/hybrid', {
  query: 'phòng trọ',
  city: 'Hà Nội',
  limit: 100
});

// Get post IDs
const postIds = result.data.post_ids;

// Enrich with SQL
const posts = await postRepository.findByIds(postIds);
```

## Troubleshooting

### Service won't start

Check Milvus connection:
```bash
docker exec -it search-service python -c "
from pymilvus import connections
connections.connect(host='milvus-standalone', port='19530')
print('Connected!')
"
```

### Slow searches

1. Check if Milvus indexes are loaded:
```python
from pymilvus import Collection
collection = Collection('posts_hybrid')
collection.load()
```

2. Monitor search time in response:
```json
{
  "search_time_ms": 450.2  // If > 500ms, investigate
}
```

3. Use GPU for faster embeddings:
```bash
DEVICE=cuda docker compose up search-service
```

## Development

### Project Structure

```
search-service/
├── config/
│   └── settings.py          # Configuration
├── services/
│   ├── embedding_service.py # BGE-M3 embeddings
│   └── search_service.py    # Hybrid search logic
├── api/
│   ├── routes.py            # FastAPI routes
│   └── schemas.py           # Pydantic models
├── app.py                   # Main application
└── requirements.txt         # Dependencies
```

### Adding New Filters

1. Update `SearchRequest` schema in `api/schemas.py`
2. Add filter logic in `search_service.py`:
```python
if new_filter:
    filters.append(f'new_field == "{new_filter}"')
```
3. Ensure field exists in Milvus collection

## License

Part of Tro Tot VN project.

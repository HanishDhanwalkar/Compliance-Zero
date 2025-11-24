import redis
from redis.commands.search.field import TextField, VectorField, NumericField, TagField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query
import numpy as np
import json
import time
from typing import List, Optional, Dict, Any
import ollama

from src.config import (
    REDIS_HOST,
    REDIS_PORT,
    OLLAMA_EMBEDD_MODEL,
    VECTOR_INDEX_NAME,
    VECTOR_DIM
)

r = redis.Redis(
    host=REDIS_HOST, 
    port=REDIS_PORT, 
    db=0, 
    decode_responses=False
)

def store_summary(user_id: str, summary: str):
    key = f"user:{user_id}:summaries"
    r.rpush(
        key, 
        summary.encode('utf-8') if isinstance(summary, str) else summary
    )
    # r.ltrim(key, -5, -1)   # keep last 5


def load_summaries(user_id: str):
    key = f"user:{user_id}:summaries"
    data = r.lrange(key, -5, -1)
    return [d.decode("utf-8") for d in data]


# ============================================================================
# 1. Semantic Caching (Vector Store) for Regulatory Documents
# ============================================================================

def _get_embedding(text: str) -> List[float]:
    """Generate embedding vector for text using Ollama."""
    response = ollama.embeddings(
        model=OLLAMA_EMBEDD_MODEL, 
        prompt=text
    )
    return response['embedding']


def initialize_vector_index():
    """Initialize Redis Search index for vector similarity search."""
    try:
        # Check if index already exists
        r.ft(VECTOR_INDEX_NAME).info()
        return True
    except:
        # Create index with vector field
        schema = (
            TextField("doc_id"),
            TextField("document_name"),
            TextField("content"),
            VectorField(
                "embedding",
                "FLAT",
                {
                    "TYPE": "FLOAT32",
                    "DIM": VECTOR_DIM,
                    "DISTANCE_METRIC": "COSINE"
                }
            )
        )
        r.ft(VECTOR_INDEX_NAME).create_index(
            schema,
            definition=IndexDefinition(
                prefix=["regulatory_doc:"], 
                index_type=IndexType.HASH
            )
        )
        return True


def store_document(doc_id: str, document_name: str, content: str):
    """
    Store a regulatory document with its vector embedding in Redis.
    
    Args:
        doc_id: Unique identifier for the document (e.g., "dodd-frank-2024")
        document_name: Human-readable name (e.g., "Dodd-Frank Act 2024")
        content: The full text content of the regulatory document
    """
    initialize_vector_index()
    
    # Generate embedding
    embedding = _get_embedding(content)
    
    # Store in Redis with vector field
    doc_key = f"regulatory_doc:{doc_id}"
    r.hset(
        doc_key, 
        mapping={
            "doc_id": doc_id,
            "document_name": document_name,
            "content": content,
            "embedding": np.array(embedding, dtype=np.float32).tobytes()
        }
    )


def search_similar_regulations(query_text: str, top_k: int = 5) -> List[Dict[str, Any]]:
    """
    Search for similar regulatory documents using semantic similarity.
    
    Args:
        query_text: The query/question to search for
        top_k: Number of top results to return
        
    Returns:
        List of dictionaries with doc_id, document_name, content, and score
    """
    initialize_vector_index()
    
    # Generate query embedding
    query_embedding = _get_embedding(query_text)
    
    # Convert to bytes for Redis Search
    query_vector = np.array(query_embedding, dtype=np.float32).tobytes()
    
    # Perform vector similarity search
    try:
        q = Query(f"*=>[KNN {top_k} @embedding $vec AS score]")\
            .sort_by("score")\
            .return_fields("doc_id", "document_name", "content", "score")\
            .dialect(2)
        
        results = r.ft(VECTOR_INDEX_NAME).search(q, query_params={"vec": query_vector})
        
        similar_docs = []
        for doc in results.docs:
            similar_docs.append({
                "doc_id": doc.doc_id,
                "document_name": doc.document_name,
                "content": doc.content,
                "score": float(doc.score) if hasattr(doc, 'score') else 0.0
            })
        return similar_docs
    except Exception as e:
        print(f"Error in vector search: {e}")
        return []


# ============================================================================
# 2. Stateful "Short-Term" Memory for Agent Thought Process
# ============================================================================

def store_thought(user_id: str, context: str, thought: str, metadata: Optional[Dict] = None):
    """
    Store an intermediate thought/finding in the agent's thought process.
    
    Args:
        user_id: User/session identifier
        context: Context of the research (e.g., "researching_company:AcmeCorp")
        thought: The intermediate finding or thought
        metadata: Optional metadata (e.g., {"source": "api", "timestamp": ...})
    """
    key = f"user:{user_id}:thoughts:{context}"
    thought_data = {
        "thought": thought,
        "timestamp": time.time(),
        "metadata": metadata or {}
    }
    r.rpush(key, json.dumps(thought_data))
    # Keep last 50 thoughts per context
    r.ltrim(key, -50, -1)


def load_thoughts(user_id: str, context: str, limit: int = 20) -> List[Dict[str, Any]]:
    """
    Load the agent's thought process for a specific context.
    
    Args:
        user_id: User/session identifier
        context: Context to retrieve thoughts for
        limit: Maximum number of thoughts to retrieve
        
    Returns:
        List of thought dictionaries with thought, timestamp, and metadata
    """
    key = f"user:{user_id}:thoughts:{context}"
    data = r.lrange(key, -limit, -1)
    thoughts = []
    for d in data:
        try:
            thoughts.append(json.loads(d.decode("utf-8")))
        except:
            pass
    return thoughts


def get_all_contexts(user_id: str) -> List[str]:
    """
    Get all active research contexts for a user.
    
    Args:
        user_id: User/session identifier
        
    Returns:
        List of context strings
    """
    pattern = f"user:{user_id}:thoughts:*"
    keys = [k.decode("utf-8") for k in r.keys(pattern)]
    # Extract context from key pattern
    contexts = []
    for key in keys:
        context = key.split(":thoughts:")[-1]
        contexts.append(context)
    return contexts


def clear_thoughts(user_id: str, context: Optional[str] = None):
    """
    Clear thoughts for a specific context or all contexts.
    
    Args:
        user_id: User/session identifier
        context: Specific context to clear, or None to clear all
    """
    if context:
        key = f"user:{user_id}:thoughts:{context}"
        r.delete(key)
    else:
        pattern = f"user:{user_id}:thoughts:*"
        keys = r.keys(pattern)
        if keys:
            r.delete(*keys)


# ============================================================================
# 3. Rate Limiting and Queue Management for External APIs
# ============================================================================

def check_rate_limit(api_name: str, max_requests: int = 100, window_seconds: int = 3600) -> bool:
    """
    Check if an API call is allowed based on rate limits.
    
    Args:
        api_name: Name of the API (e.g., "sec_gov", "fincen")
        max_requests: Maximum requests allowed in the time window
        window_seconds: Time window in seconds (default 1 hour)
        
    Returns:
        True if request is allowed, False if rate limit exceeded
    """
    key = f"rate_limit:{api_name}"
    current = r.incr(key)
    
    if current == 1:
        # First request in this window, set expiration
        r.expire(key, window_seconds)
    
    return current <= max_requests


def get_rate_limit_status(api_name: str) -> Dict[str, Any]:
    """
    Get current rate limit status for an API.
    
    Args:
        api_name: Name of the API
        
    Returns:
        Dictionary with current count, limit, and time remaining
    """
    key = f"rate_limit:{api_name}"
    current = int(r.get(key) or 0)
    ttl = r.ttl(key)
    
    return {
        "api_name": api_name,
        "current_requests": current,
        "time_remaining": ttl if ttl > 0 else 0
    }


def enqueue_api_request(api_name: str, request_data: Dict[str, Any], priority: int = 0):
    """
    Add an API request to the queue for processing.
    
    Args:
        api_name: Name of the API
        request_data: Dictionary containing request details
        priority: Priority level (higher = more priority, default 0)
    """
    queue_key = f"api_queue:{api_name}"
    request = {
        "data": request_data,
        "priority": priority,
        "timestamp": time.time()
    }
    # Use sorted set for priority queue
    r.zadd(queue_key, {json.dumps(request): priority})


def dequeue_api_request(api_name: str) -> Optional[Dict[str, Any]]:
    """
    Get the next API request from the queue (highest priority first).
    
    Args:
        api_name: Name of the API
        
    Returns:
        Request data dictionary or None if queue is empty
    """
    queue_key = f"api_queue:{api_name}"
    # Get highest priority item (highest score)
    result = r.zrevrange(queue_key, 0, 0, withscores=True)
    
    if result:
        request_json, _ = result[0]
        request = json.loads(request_json.decode("utf-8"))
        # Remove from queue
        r.zrem(queue_key, request_json)
        return request["data"]
    return None


def get_queue_size(api_name: str) -> int:
    """
    Get the current size of the API request queue.
    
    Args:
        api_name: Name of the API
        
    Returns:
        Number of requests in queue
    """
    queue_key = f"api_queue:{api_name}"
    return r.zcard(queue_key)


def clear_queue(api_name: str):
    """
    Clear all requests from an API queue.
    
    Args:
        api_name: Name of the API
    """
    queue_key = f"api_queue:{api_name}"
    r.delete(queue_key)
import redis
from redis.commands.search.index_definition  import IndexDefinition, IndexType
from redis.commands.search.field import TextField, VectorField

from typing import List, Dict, Any

from src.embedding import get_embeddings

from src.config import (
    REDIS_URL,
    VECTOR_DIM
)

print("connecting to redis...[to: ", REDIS_URL, "]")
r = redis.Redis.from_url(
    url=REDIS_URL,
)

def create_or_update_index(index_name="idx:docs", prefix="doc:"):    
    try:
        print("Creating index...")
        fields = [
            TextField("title"),
            TextField("source_url"),
            TextField("text"),

            # VECTOR field (HNSW)
            VectorField(
                "embedding",
                "HNSW",
                {
                    "type": "FLOAT32",
                    "dim": VECTOR_DIM,
                    "distance_metric": "COSINE",
                    "initial_cap": 1000,
                    "m": 16,
                    "ef_construction": 200,
                }
            )
        ]

        # Define index
        definition = IndexDefinition(
            prefix=[prefix],
            index_type=IndexType.HASH
        )

        # Create index
        r.ft(index_name).create_index(
            fields=fields,
            definition=definition
        )

        print("Index created.")

        print("Index created.")
    except redis.ResponseError as e:
        if "Index already exists" in str(e):
            print("Index already exists, skipping.")
        else:
            raise

def upsert_doc_chunk(doc_id: str, chunk_id: int, title: str, text: str, source_url: str, page: int = None):
    key = f"doc:{doc_id}:chunk:{chunk_id}"

    emb = get_embeddings(text)

    mapping = {
        "title": title,
        "doc_id": doc_id,
        "chunk_id": str(chunk_id),
        "text": text,
        "source_url": source_url,
        "page": str(page) if page is not None else "",
        "embedding": emb.tobytes(),
    }

    r.hset(key, mapping=mapping)
    return key

def semantic_search(
    query: str, 
    top_k: int = 3, 
    index_name:str ="idx:docs"
) -> Any:
    q_emb = get_embeddings(query)

    knn_query = f"*=>[KNN {top_k} @embedding $vec_param AS score]" 
    args = [
        "FT.SEARCH", index_name, knn_query,
        "PARAMS", "2", "vec_param", q_emb.tobytes(),
        "SORTBY", "score",
        "RETURN", "6", 
            "title", 
            "text", 
            "source_url", 
            "doc_id", 
            "chunk_id", 
            "page",
        "DIALECT", "2"
    ]

    resp = r.execute_command(*args)
    return resp

# file: test_redis_helpers.py

import redis
from src.config import VECTOR_DIM
from src.embedding import get_embeddings
from src.redis_tools import (
    r,
    create_or_update_index,
    upsert_doc_chunk,
    semantic_search
)

def test_redis_connection():
    print("\n--- test_redis_connection ---")
    try:
        pong = r.ping()
        print("Redis PING:", pong)
        assert pong is True
    except Exception as e:
        print("Redis connection failed:", e)
        raise


def test_index_creation():
    print("\n--- test_index_creation ---")
    try:
        create_or_update_index("idx:docs")
        print("Index creation executed successfully.")
    except Exception as e:
        print("Index creation failed:", e)
        raise


def test_embedding_call():
    print("\n--- test_embedding_call ---")
    try:
        vector = get_embeddings("hello world")
        print("Embedding length:", len(vector))
        assert len(vector) > 0 and len(vector) == VECTOR_DIM
    except Exception as e:
        print("Embedding failed:", e)
        raise


def test_upsert_chunk():
    print("\n--- test_upsert_chunk ---")
    try:
        key = upsert_doc_chunk(
            doc_id="demo1",
            chunk_id=0,
            title="Test Doc",
            text="This is a test chunk for Redis.",
            source_url="local"
        )
        print("Upserted key:", key)

        exists = r.exists(key)
        print("Key exists:", exists)
        assert exists == 1
    except Exception as e:
        print("Upsert failed:", e)
        raise


def test_semantic_search():
    print("\n--- test_semantic_search ---")
    try:
        results = semantic_search("redis test", top_k=3)
        print("Search results:", results)

        assert isinstance(results, list)
    except Exception as e:
        print("Search failed:", e)
        raise


if __name__ == "__main__":
    print("\n===== RUNNING BASIC REDIS/OLLAMA TESTS =====")
    test_redis_connection()
    test_index_creation()
    test_embedding_call()
    test_upsert_chunk()
    test_semantic_search()
    print("\n===== ALL TESTS DONE =====")

OLLAMA_LLM_MODEL="llama3.2:latest" 
OLLAMA_EMBEDD_MODEL="mxbai-embed-large:latest" 


REDIS_HOST="localhost"
REDIS_PORT=6379

VECTOR_INDEX_NAME = "regulatory_docs_idx"
VECTOR_DIM = 1024  # mxbai-embed-large produces 1024-dimensional vectors

THOUGHTS_TO_STORE_LIMIT = 10
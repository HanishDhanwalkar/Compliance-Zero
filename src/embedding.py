import numpy as np

import ollama
from src.config import (
    VECTOR_DIM, OLLAMA_EMBEDD_MODEL
)

def get_embeddings(text: str) -> list[float]:
    resp = ollama.embed(
        model=OLLAMA_EMBEDD_MODEL, 
        input=text
    )
    vec = np.array(resp.embeddings[0], dtype=np.float32)
    assert vec.shape[0] == VECTOR_DIM
    return vec

if __name__ == "__main__":
    print(get_embeddings("Hello world!"))
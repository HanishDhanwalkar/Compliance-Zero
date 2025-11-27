from ollama import chat
from termcolor import colored

from src.redis_tools import (
    semantic_search,
    log_query
)
from src.risk_rule_engine import (
    run_risk_rules
)
from src.config import (
    INDEX_NAME,
    OLLAMA_LLM_MODEL
)

def rag_answer(query):
    retrieved = semantic_search(query, 3, INDEX_NAME)
    print(colored(f"Retrieved:\n{retrieved}", "green"))

    context = "\n\n".join(
        f"[{i}] {item[b'text']}\n(source: {item[b'source_url']} chunk={item[b'chunk_id']})"
        for i, item in enumerate(retrieved)
    )

    prompt = f"""
You are a FinTech compliance assistant.

User query:
{query}

Relevant documents:
{context}

Answer clearly with citations in format:
[CITATION: source_url | chunk_id]

If unsure, say so.
"""

    resp = chat(
        model=OLLAMA_LLM_MODEL,
        messages=[
            {
                "role":"user", 
                "content": prompt
            }
        ]
    )
    answer = resp["message"]["content"]
    print(colored(f"LLM answer:\n{answer}", "green"))

    risk_flags = run_risk_rules(answer)
    log_query(query, retrieved, answer, risk_flags)

    return {
        "answer": answer,
        "citations": retrieved,
        "risk_flags": risk_flags
    }

from src.rag import rag_answer

print("TEST #1")
result = rag_answer("What sanctions apply here?")
print(result)
assert "answer" in result
assert "citations" in result
assert "risk_flags" in result
print("Test 1 passed: structure OK")


# 2. Citation correctness test
print("TEST #2")
assert result["citations"][0]["source_url"] == "doc1.pdf"
assert result["citations"][0]["chunk_id"] == 1
print("Test 2 passed: citations OK")


# 3. Risk flags detection test
print("TEST #3")
assert "sanctions" in result["risk_flags"]
print("Test 3 passed: risk flag detection OK")

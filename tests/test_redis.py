from src.redis_tools import *
from datetime import datetime

def test_store_summaries():
    store_summary(
        "test_user", 
        f"test summary @ time: {datetime.now().strftime('%d %H:%M')}"
    )
    

def test_load_summaries():
    summaries = load_summaries("test_user")
    
    for summary in summaries:
        assert isinstance(summary, str)
        print(summary)
        
def test_store_docs():
    store_document(
        f"doc_id_{datetime.now().strftime('%d %H:%M')}",
        "test_doc", 
        "test_doc_content"
    )

def test_search_docs():
    search_similar_regulations(
        query_text="test_query_at_time: " + datetime.now().strftime('%d %H:%M'),
    )
        
        
if __name__ == "__main__":
    # test_store_summaries()
    # test_load_summaries()
    test_store_docs()
    test_search_docs()
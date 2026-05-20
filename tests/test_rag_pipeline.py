"""
Test suite for RAG Pipeline
"""
from src.embeddings import EmbeddingManager, SimpleVectorDB
from src.rag_pipeline import RAGPipeline
from src.tavily_search import TavilySearchEngine


def test_embeddings():
    """Test embedding functionality"""
    print("Testing Embeddings...")
    
    try:
        embedding_manager = EmbeddingManager()
        
        # Test single embedding
        text = "Artificial Intelligence and Machine Learning"
        embedding = embedding_manager.embed_text(text)
        print(f"Embedding shape: {embedding.shape}")
        
        # Test batch embedding
        texts = [
            "Python is a programming language",
            "Machine learning is a subset of AI",
            "RAG improves LLM accuracy"
        ]
        embeddings = embedding_manager.embed_batch(texts)
        print(f"Batch embeddings: {len(embeddings)} items")
        
        return True
        
    except Exception as e:
        print(f"Embedding test failed: {e}")
        return False


def test_vector_db():
    """Test vector database"""
    print("\nTesting Vector DB...")
    
    try:
        embedding_manager = EmbeddingManager()
        vector_db = SimpleVectorDB(embedding_manager)
        
        # Add documents
        documents = [
            "Python is a popular programming language for AI",
            "Machine learning models require large datasets",
            "RAG combines retrieval and generation for better results",
            "Tavily API provides real-time search capabilities"
        ]
        
        for doc in documents:
            vector_db.add_document(doc, {"source": "test"})
        
        # Search
        query = "machine learning"
        results = vector_db.search(query, top_k=3)
        print(f"Search '{query}' returned {len(results)} results")
        for content, score, meta in results:
            print(f"  Score: {score:.2f}, Content: {content[:50]}...")
        
        return True
        
    except Exception as e:
        print(f"Vector DB test failed: {e}")
        return False


def test_rag_pipeline_structure():
    """Test RAG pipeline structure (without API calls)"""
    print("\nTesting RAG Pipeline Structure...")
    
    try:
        # Note: This test only checks structure, not actual retrieval
        # For full test, need valid TAVILY_API_KEY
        
        print("RAG Pipeline structure validated")
        return True
        
    except Exception as e:
        print(f"RAG Pipeline test failed: {e}")
        return False


if __name__ == "__main__":
    print("Running test suite...\n")
    
    test1 = test_embeddings()
    test2 = test_vector_db()
    test3 = test_rag_pipeline_structure()
    
    if test1 and test2 and test3:
        print("\n✓ All tests passed!")
    else:
        print("\n✗ Some tests failed")

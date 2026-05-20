"""
Integration tests using Mock Tavily
No API key required - perfect for development and testing
"""
import asyncio
from src.mock_tavily import MockTavilySearchEngine
from src.search_factory import create_search_engine
from src.embeddings import EmbeddingManager, SimpleVectorDB
from src.rag_pipeline import RAGPipeline


async def test_mock_tavily_search():
    """Test mock Tavily search"""
    print("\n" + "=" * 60)
    print("Test 1: Mock Tavily Search")
    print("=" * 60)
    
    try:
        engine = MockTavilySearchEngine()
        
        # Test searches
        queries = [
            "RAG",
            "machine learning",
            "embeddings"
        ]
        
        for query in queries:
            print(f"\n📝 Query: {query}")
            results = await engine.search(query, max_results=3)
            print(f"Results: {len(results)} items")
            
            for i, result in enumerate(results, 1):
                print(f"  {i}. {result.title} (Score: {result.score:.2f})")
        
        print("\n✓ Mock Tavily test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Mock Tavily test failed: {e}")
        return False


async def test_search_factory():
    """Test search factory (mock vs real)"""
    print("\n" + "=" * 60)
    print("Test 2: Search Factory")
    print("=" * 60)
    
    try:
        # Force mock
        search_engine = create_search_engine(use_mock=True)
        print(f"✓ Created search engine: {type(search_engine).__name__}")
        
        # Test search
        results = await search_engine.search("Python programming")
        print(f"✓ Search returned {len(results)} results")
        
        return True
        
    except Exception as e:
        print(f"✗ Factory test failed: {e}")
        return False


async def test_rag_pipeline_with_mock():
    """Test RAG pipeline with mock search"""
    print("\n" + "=" * 60)
    print("Test 3: RAG Pipeline with Mock")
    print("=" * 60)
    
    try:
        # Create components
        search_engine = create_search_engine(use_mock=True)
        embedding_manager = EmbeddingManager()
        vector_db = SimpleVectorDB(embedding_manager)
        
        # Add documents
        docs = [
            "Python is a programming language",
            "Machine learning uses algorithms",
            "RAG improves LLM accuracy",
            "Embeddings are numerical representations"
        ]
        vector_db.add_documents(docs)
        print(f"✓ Added {len(docs)} documents to vector DB")
        
        # Create RAG pipeline
        rag = RAGPipeline(search_engine, vector_db, 
                         use_query_expansion=True,
                         use_reranking=True)
        
        # Test retrieval
        query = "How does machine learning work?"
        print(f"\n📝 Query: {query}")
        
        results = await rag.retrieve(query, top_k=5)
        print(f"✓ Retrieved {len(results)} results")
        
        for i, result in enumerate(results, 1):
            print(f"  {i}. [{result.source}] Score: {result.score:.2f}")
            print(f"     {result.content[:70]}...")
        
        print("\n✓ RAG Pipeline test passed!")
        return True
        
    except Exception as e:
        print(f"✗ RAG Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_batch_search():
    """Test batch search with mock"""
    print("\n" + "=" * 60)
    print("Test 4: Batch Search")
    print("=" * 60)
    
    try:
        engine = MockTavilySearchEngine()
        
        queries = ["RAG", "embeddings", "vector database"]
        results = await engine.batch_search(queries)
        
        print(f"✓ Searched {len(queries)} queries")
        for query, items in results.items():
            print(f"  - '{query}': {len(items)} results")
        
        print("\n✓ Batch search test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Batch search test failed: {e}")
        return False


async def run_all_tests():
    """Run all integration tests"""
    print("\n" + "=" * 70)
    print("MOCK TAVILY INTEGRATION TESTS - No API Key Required!")
    print("=" * 70)
    
    tests = [
        ("Mock Tavily Search", test_mock_tavily_search),
        ("Search Factory", test_search_factory),
        ("RAG Pipeline with Mock", test_rag_pipeline_with_mock),
        ("Batch Search", test_batch_search)
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 70)
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)

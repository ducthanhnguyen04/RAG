"""
Test suite for Tavily Search API
"""
import asyncio
from src.tavily_search import TavilySearchEngine, SearchResult


async def test_tavily_search():
    """Test Tavily search functionality"""
    # Note: Requires TAVILY_API_KEY in environment
    
    try:
        engine = TavilySearchEngine()
        
        # Test 1: Basic search
        print("Test 1: Basic search")
        results = await engine.search("Python machine learning libraries", max_results=5)
        print(f"Found {len(results)} results")
        for result in results[:2]:
            print(f"  - {result.title}: {result.score:.2f}")
        
        # Test 2: Search with direct answer
        print("\nTest 2: Search with answer")
        results = await engine.search("What is RAG?", include_answer=True)
        print(f"Found {len(results)} results")
        
        # Test 3: Batch search
        print("\nTest 3: Batch search")
        queries = [
            "Tavily API documentation",
            "RAG implementation",
            "LLM optimization"
        ]
        batch_results = await engine.batch_search(queries)
        for query, results in batch_results.items():
            print(f"  '{query}': {len(results)} results")
        
        print("\nAll tests passed!")
        return True
        
    except Exception as e:
        print(f"Test failed: {e}")
        return False


if __name__ == "__main__":
    success = asyncio.run(test_tavily_search())
    exit(0 if success else 1)

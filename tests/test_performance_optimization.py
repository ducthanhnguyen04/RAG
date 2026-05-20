"""
Test Performance Optimization Features
Tests caching, batch processing, and performance tuning
"""
import asyncio
import time
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.caching import MemoryCache, FileCache, HybridCache, CacheManager
from src.batch_processing import BatchProcessor, QueryGrouper, ProgressTracker, CostOptimizer
from src.performance_tuning import PerformanceTuner, LatencyOptimizer, ResourceOptimizer
from src.advanced_rag import AdvancedRAGSystem
from utils.logger import setup_logger

logger = setup_logger(__name__)


def test_memory_cache():
    """Test memory cache functionality"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Memory Cache")
    logger.info("="*70)
    
    cache = MemoryCache(max_size=100)
    
    # Test basic operations
    cache.set("key1", "value1", ttl=3600)
    result = cache.get("key1")
    assert result == "value1", "Cache get failed"
    logger.info("✓ Basic get/set works")
    
    # Test hit tracking
    cache.get("key1")
    cache.get("key1")
    stats = cache.get_stats()
    assert stats["hits"] == 2, "Hit tracking failed"
    logger.info(f"✓ Hit tracking works: {stats['hits']} hits")
    
    # Test expiration
    cache.set("key2", "value2", ttl=1)  # 1 second TTL
    time.sleep(1.5)
    result = cache.get("key2")
    assert result is None, "Expiration failed"
    logger.info("✓ TTL expiration works")
    
    logger.info("✓ Memory cache test PASSED\n")


def test_file_cache():
    """Test file cache functionality"""
    logger.info("\n" + "="*70)
    logger.info("TEST: File Cache")
    logger.info("="*70)
    
    cache = FileCache()
    
    # Test basic operations
    cache.set("key1", {"data": "test"}, ttl=3600)
    result = cache.get("key1")
    assert result["data"] == "test", "File cache get failed"
    logger.info("✓ File-based persistence works")
    
    # Verify file exists
    cache_files = list(Path(".cache").glob("*"))
    assert len(cache_files) > 0, "Cache file not created"
    logger.info(f"✓ Cache file created: {cache_files[0].name}")
    
    logger.info("✓ File cache test PASSED\n")


def test_hybrid_cache():
    """Test hybrid cache functionality"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Hybrid Cache")
    logger.info("="*70)
    
    cache = HybridCache(memory_size=100)
    
    # Test memory layer
    cache.set("key1", "value1", ttl=3600, layer="memory")
    result = cache.get("key1", layer="memory")
    assert result == "value1", "Memory layer failed"
    logger.info("✓ Memory layer works")
    
    # Test file layer
    cache.set("key2", {"data": "persistent"}, ttl=3600, layer="file")
    result = cache.get("key2", layer="file")
    assert result["data"] == "persistent", "File layer failed"
    logger.info("✓ File layer works")
    
    # Test auto-promotion
    cache.set("key3", "value3", ttl=3600)  # Default layer
    assert cache.get("key3") is not None, "Auto-set failed"
    logger.info("✓ Hybrid auto-layering works")
    
    logger.info("✓ Hybrid cache test PASSED\n")


def test_performance_tuner():
    """Test performance tuning recommendations"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Performance Tuner")
    logger.info("="*70)
    
    tuner = PerformanceTuner()
    
    # Test tuning ranges
    ranges = tuner.get_tuning_ranges()
    assert "chunk_size" in ranges, "Missing tuning parameter"
    logger.info(f"✓ Tuning ranges available: {len(ranges)} parameters")
    
    # Test configuration recommendations
    latency_config = tuner.recommend_config_for_latency(target_latency_ms=500)
    assert latency_config["top_k_retrieval"] == 3, "Latency config incorrect"
    logger.info("✓ Low-latency config recommended")
    
    quality_config = tuner.recommend_config_for_quality()
    assert quality_config["top_k_retrieval"] == 10, "Quality config incorrect"
    logger.info("✓ High-quality config recommended")
    
    balanced_config = tuner.recommend_config_balanced()
    assert balanced_config["top_k_retrieval"] == 5, "Balanced config incorrect"
    logger.info("✓ Balanced config recommended")
    
    # Test memory estimation
    memory_est = tuner.estimate_memory_usage(
        num_documents=1000,
        chunk_size=512
    )
    assert memory_est["total_mb"] > 0, "Memory estimation failed"
    logger.info(f"✓ Memory estimation: {memory_est['total_mb']:.2f} MB for 1000 docs")
    
    # Test cost estimation
    cost_est = tuner.estimate_cost_per_query(
        use_tavily=True,
        use_openai=True,
        queries=1000
    )
    assert "cost_per_query" in cost_est, "Cost estimation failed"
    logger.info(f"✓ Cost per query: ${cost_est['cost_per_query']:.6f}")
    
    logger.info("✓ Performance tuner test PASSED\n")


def test_latency_optimizer():
    """Test latency optimization analysis"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Latency Optimizer")
    logger.info("="*70)
    
    optimizer = LatencyOptimizer()
    
    # Simulate latency breakdown
    latency_breakdown = {
        "query_optimization": 50,
        "retrieval": 800,
        "generation": 300
    }
    
    analysis = optimizer.analyze_bottlenecks(latency_breakdown)
    assert len(analysis) > 0, "Analysis failed"
    logger.info(f"✓ Analyzed {len(analysis)} components")
    
    for component, latency, recommendation in analysis:
        logger.info(f"  {component}: {latency:.0f}ms → {recommendation}")
    
    logger.info("✓ Latency optimizer test PASSED\n")


def test_resource_optimizer():
    """Test resource optimization tips"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Resource Optimizer")
    logger.info("="*70)
    
    optimizer = ResourceOptimizer()
    
    tips = optimizer.get_optimization_tips()
    assert len(tips) > 0, "No tips returned"
    logger.info(f"✓ Got {len(tips)} optimization tips:")
    
    for i, tip in enumerate(tips, 1):
        logger.info(f"  {i}. {tip}")
    
    logger.info("✓ Resource optimizer test PASSED\n")


def test_query_grouper():
    """Test query grouping functionality"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Query Grouper")
    logger.info("="*70)
    
    grouper = QueryGrouper()
    
    queries = [
        "What is RAG?",
        "How does RAG work?",
        "Explain RAG architecture",
        "What is machine learning?",
        "How to use embeddings?"
    ]
    
    # Test intent grouping
    grouped = grouper.group_by_intent(queries)
    logger.info(f"✓ Grouped {len(queries)} queries into {len(grouped)} groups")
    
    for intent, group_queries in grouped.items():
        logger.info(f"  {intent}: {len(group_queries)} queries")
    
    logger.info("✓ Query grouper test PASSED\n")


def test_cost_optimizer():
    """Test cost optimization"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Cost Optimizer")
    logger.info("="*70)
    
    optimizer = CostOptimizer()
    
    # Test cost estimation
    cost = optimizer.estimate_cost(
        queries=100,
        avg_tokens=500,
        use_tavily=True,
        use_openai=True
    )
    
    assert cost > 0, "Cost estimation failed"
    logger.info(f"✓ Cost for 100 queries: ${cost:.4f}")
    
    # Test optimization recommendations
    recommendations = optimizer.get_optimization_recommendations()
    assert len(recommendations) > 0, "No recommendations returned"
    logger.info(f"✓ Got {len(recommendations)} recommendations")
    
    logger.info("✓ Cost optimizer test PASSED\n")


async def test_advanced_rag_with_caching():
    """Test Advanced RAG with caching integration"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Advanced RAG with Caching")
    logger.info("="*70)
    
    # Initialize system with caching
    system = AdvancedRAGSystem(
        use_mock_search=True,
        use_mock_llm=True,
        enable_caching=True,
        cache_ttl=3600,
        cache_strategy="hybrid"
    )
    
    logger.info("✓ System initialized with caching enabled")
    
    # Test first query (cache miss)
    logger.info("\nFirst query (expected cache miss)...")
    start = time.time()
    answer1 = await system.answer("What is RAG?", top_k=3)
    latency1 = time.time() - start
    logger.info(f"  Latency: {latency1*1000:.0f}ms")
    
    # Test second query (cache hit)
    logger.info("\nSecond query - same question (expected cache hit)...")
    start = time.time()
    answer2 = await system.answer("What is RAG?", top_k=3)
    latency2 = time.time() - start
    logger.info(f"  Latency: {latency2*1000:.0f}ms")
    
    # Cache hit should be faster
    speedup = latency1 / latency2 if latency2 > 0 else 1
    logger.info(f"  Speedup: {speedup:.1f}x")
    
    # Check cache stats
    cache_stats = system.get_cache_stats()
    logger.info(f"✓ Cache stats: {cache_stats}")
    
    logger.info("✓ Advanced RAG caching test PASSED\n")


async def test_batch_processing():
    """Test batch processing functionality"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Batch Processing")
    logger.info("="*70)
    
    system = AdvancedRAGSystem(
        use_mock_search=True,
        use_mock_llm=True,
        enable_caching=False
    )
    
    queries = [
        "What is RAG?",
        "How does embedding work?",
        "What is vector search?",
        "Explain Tavily API"
    ]
    
    logger.info(f"Processing {len(queries)} queries in batch...")
    
    start = time.time()
    results = await system.batch_answer(queries, top_k=3, use_grouping=True)
    total_time = time.time() - start
    
    assert len(results) == len(queries), "Not all queries processed"
    logger.info(f"✓ Successfully processed {len(results)} queries")
    logger.info(f"✓ Total time: {total_time*1000:.0f}ms")
    logger.info(f"✓ Average per query: {(total_time*1000/len(queries)):.0f}ms")
    
    logger.info("✓ Batch processing test PASSED\n")


async def test_performance_recommendations():
    """Test performance recommendations"""
    logger.info("\n" + "="*70)
    logger.info("TEST: Performance Recommendations")
    logger.info("="*70)
    
    system = AdvancedRAGSystem(
        use_mock_search=True,
        use_mock_llm=True
    )
    
    # Process a few queries to gather metrics
    queries = ["What is RAG?", "How does search work?"]
    for query in queries:
        await system.answer(query, top_k=3)
    
    # Get recommendations
    recommendations = system.get_tuning_recommendations()
    
    logger.info("✓ Generated tuning recommendations:")
    logger.info(f"  Latency config: {recommendations['low_latency_config']['notes']}")
    logger.info(f"  Quality config: {recommendations['high_quality_config']['notes']}")
    logger.info(f"  Balanced config: {recommendations['balanced_config']['notes']}")
    logger.info(f"  Tips: {len(recommendations['optimization_tips'])} optimization tips")
    
    logger.info("✓ Performance recommendations test PASSED\n")


def main():
    """Run all tests"""
    logger.info("\n" + "="*70)
    logger.info("PERFORMANCE OPTIMIZATION TEST SUITE")
    logger.info("="*70)
    
    try:
        # Synchronous tests
        test_memory_cache()
        test_file_cache()
        test_hybrid_cache()
        test_performance_tuner()
        test_latency_optimizer()
        test_resource_optimizer()
        test_query_grouper()
        test_cost_optimizer()
        
        # Async tests
        asyncio.run(test_advanced_rag_with_caching())
        asyncio.run(test_batch_processing())
        asyncio.run(test_performance_recommendations())
        
        logger.info("\n" + "="*70)
        logger.info("✅ ALL TESTS PASSED!")
        logger.info("="*70 + "\n")
        
    except Exception as e:
        logger.error(f"\n❌ TEST FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()

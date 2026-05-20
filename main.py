#!/usr/bin/env python3
"""
RAG + Tavily Search API - Advanced RAG System
Main Entry Point with Full Pipeline + Performance Optimization
"""
import asyncio
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from config.settings import TAVILY_API_KEY, USE_MOCK_TAVILY, LLM_LANGUAGE
from src.advanced_rag import AdvancedRAGSystem
from utils.logger import setup_logger

logger = setup_logger(__name__)


async def demo_basic_retrieval(rag_system):
    """Demo basic retrieval with caching"""
    logger.info("\n" + "="*70)
    logger.info("DEMO 1: Basic Retrieval with Caching")
    logger.info("="*70)
    
    query = "What is Retrieval-Augmented Generation?"
    
    # First query (cache miss)
    logger.info("\n[Query 1 - Cache MISS]")
    start = time.time()
    answer = await rag_system.answer(query, top_k=3)
    latency1 = time.time() - start
    
    logger.info(f"Answer: {answer.answer[:200]}...")
    logger.info(f"Confidence: {answer.confidence:.2f}")
    logger.info(f"Latency: {answer.latency_ms:.0f}ms")
    
    # Second query (cache hit)
    logger.info(f"\n[Query 2 - Same query (Cache HIT expected)]")
    start = time.time()
    answer = await rag_system.answer(query, top_k=3)
    latency2 = time.time() - start
    
    logger.info(f"Latency: {answer.latency_ms:.0f}ms")
    if latency2 < latency1 * 0.8:
        logger.info(f"✓ Cache speedup: {latency1/latency2:.1f}x faster!")
    
    # Show cache stats
    cache_stats = rag_system.get_cache_stats()
    if cache_stats.get("enabled"):
        logger.info(f"\nCache Stats: {cache_stats['stats']}")


async def demo_batch_processing(rag_system):
    """Demo batch processing with grouping"""
    logger.info("\n" + "="*70)
    logger.info("DEMO 2: Batch Processing with Query Grouping")
    logger.info("="*70)
    
    queries = [
        "What is RAG?",
        "How does Tavily work?",
        "Explain embeddings",
        "What are vector databases?",
        "How to evaluate RAG systems?"
    ]
    
    logger.info(f"\nProcessing {len(queries)} queries in batch...")
    logger.info("With query grouping to reduce redundant processing...\n")
    
    start = time.time()
    results = await rag_system.batch_answer(queries, top_k=3, use_grouping=True)
    total_time = time.time() - start
    
    logger.info(f"\n✓ Results Summary:")
    logger.info(f"  Total queries: {len(queries)}")
    logger.info(f"  Processed: {len(results)}")
    logger.info(f"  Total time: {total_time*1000:.0f}ms")
    logger.info(f"  Average per query: {(total_time*1000/len(queries)):.0f}ms")
    
    for i, result in enumerate(results, 1):
        logger.info(f"\n  [{i}] {result.query[:50]}...")
        logger.info(f"      Confidence: {result.confidence:.2f}")
        logger.info(f"      Latency: {result.latency_ms:.0f}ms")


async def demo_performance_tuning(rag_system):
    """Demo performance tuning recommendations"""
    logger.info("\n" + "="*70)
    logger.info("DEMO 3: Performance Tuning Recommendations")
    logger.info("="*70)
    
    logger.info("\nGenerating tuning recommendations based on current metrics...\n")
    
    recommendations = rag_system.get_tuning_recommendations()
    
    # Low latency config
    logger.info("[1] Low-Latency Configuration (<500ms target):")
    config = recommendations['low_latency_config']
    logger.info(f"    • Chunk size: {config['chunk_size']}")
    logger.info(f"    • Top-K: {config['top_k_retrieval']}")
    logger.info(f"    • Query expansion: {config['query_expansion']}")
    logger.info(f"    • Embedding model: {config['embedding_model']}")
    logger.info(f"    • LLM tokens: {config['llm_max_tokens']}")
    
    # High quality config
    logger.info("\n[2] High-Quality Configuration:")
    config = recommendations['high_quality_config']
    logger.info(f"    • Chunk size: {config['chunk_size']}")
    logger.info(f"    • Top-K: {config['top_k_retrieval']}")
    logger.info(f"    • Query expansion: {config['query_expansion']}")
    logger.info(f"    • Embedding model: {config['embedding_model']}")
    logger.info(f"    • Reranking: {config['reranking']}")
    logger.info(f"    • LLM tokens: {config['llm_max_tokens']}")
    
    # Balanced config
    logger.info("\n[3] Balanced Configuration:")
    config = recommendations['balanced_config']
    logger.info(f"    • Chunk size: {config['chunk_size']}")
    logger.info(f"    • Top-K: {config['top_k_retrieval']}")
    logger.info(f"    • Caching: {config['cache_enabled']}")
    logger.info(f"    • Batch size: {config['batch_size']}")
    
    # Optimization tips
    logger.info("\n[4] Optimization Tips:")
    for i, tip in enumerate(recommendations['optimization_tips'][:5], 1):
        logger.info(f"    {i}. {tip}")
    logger.info(f"    ... and {len(recommendations['optimization_tips'])-5} more")


async def main():
    """Main application entry point"""
    logger.info("=" * 70)
    logger.info("ADVANCED RAG SYSTEM - Real Tavily + Query Rewriting + Filtering")
    logger.info("=" * 70)
    
    # Check if using mock or real API
    if not TAVILY_API_KEY:
        logger.error("❌ TAVILY_API_KEY not found!")
        logger.error("Please set TAVILY_API_KEY environment variable")
        sys.exit(1)
    else:
        logger.info("✓ Using Real Tavily API")
        logger.info("✓ Using Real Query Rewriting")
        logger.info("✓ Using Real Filtering & Definition Detection")
        logger.info("✓ Using MOCK LLM (for local testing)\n")
    
    try:
        # Initialize Advanced RAG System with optimization features
        logger.info("Initializing Advanced RAG System with real Tavily...\n")
        
        rag_system = AdvancedRAGSystem(
            use_mock_search=False,  # Use REAL Tavily API
            use_mock_llm=True,  # Keep mock LLM for testing
            enable_caching=True,
            cache_ttl=3600,
            cache_strategy="hybrid",
            language=LLM_LANGUAGE
        )
        
        # Add sample documents to vector DB
        logger.info("Adding sample documents...")
        sample_documents = [
            "Retrieval-Augmented Generation (RAG) is a technique that combines information retrieval with generation. It retrieves relevant documents first, then uses them as context for generating better answers.",
            "Tavily is an AI-native search engine optimized for AI agents. It provides real-time web search with advanced filtering and ranking capabilities.",
            "Large Language Models (LLMs) are neural networks trained on massive amounts of text. They can generate human-like text but sometimes hallucinate facts.",
            "Vector databases store high-dimensional embeddings for efficient similarity search. They enable semantic search and retrieval at scale.",
            "Query optimization improves retrieval quality by expanding queries, extracting keywords, and rewriting queries for better coverage.",
            "Evaluation metrics in RAG include precision, recall, F1 score, MRR, NDCG, latency, and cost. These help measure system performance.",
            "Caching is crucial for production RAG systems. It reduces redundant API calls and improves response time significantly.",
            "Batch processing allows handling multiple queries efficiently. Query grouping can further reduce computational overhead.",
        ]
        
        rag_system.add_documents(sample_documents)
        logger.info(f"✓ Added {len(sample_documents)} documents\n")
        
        # Run demos
        await demo_basic_retrieval(rag_system)
        await demo_batch_processing(rag_system)
        await demo_performance_tuning(rag_system)
        
        # Print evaluation report
        logger.info(f"\n{'='*70}")
        logger.info("EVALUATION REPORT")
        logger.info(f"{'='*70}")
        report = rag_system.get_evaluation_report()
        if report:
            logger.info(f"Queries processed: {getattr(report, 'total_queries', 0)}")
            logger.info(f"Average latency: {getattr(report, 'avg_latency_ms', 0):.0f}ms")
            logger.info(f"Average precision: {getattr(report, 'avg_precision', 0):.2f}")
        
        logger.info(f"\n{'='*70}")
        logger.info("✓ Advanced RAG System with Performance Optimization Demo Completed!")
        sys.exit(1)
    except Exception as e:
        logger.error(f"An error occurred: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())

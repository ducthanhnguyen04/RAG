"""
Test suite for Advanced RAG System
Tests all components: query optimization, retrieval, generation, evaluation
"""
import asyncio
from src.advanced_rag import AdvancedRAGSystem
from src.query_optimizer import QueryOptimizer, QueryIntent
from src.llm_generator import MockLLMGenerator
from src.evaluation import RAGEvaluator


def test_query_optimizer():
    """Test query optimization"""
    print("\n" + "="*60)
    print("Test 1: Query Optimizer")
    print("="*60)
    
    try:
        optimizer = QueryOptimizer()
        
        # Test intent detection
        queries = [
            ("What is RAG?", QueryIntent.WHAT),
            ("How to implement machine learning?", QueryIntent.HOW),
            ("Why is AI important?", QueryIntent.WHY),
        ]
        
        for query, expected_intent in queries:
            intent = optimizer.detect_intent(query)
            print(f"✓ '{query}' → {intent.value}")
            assert intent == expected_intent, f"Expected {expected_intent}, got {intent}"
        
        # Test query expansion
        query = "What is RAG?"
        variations = optimizer.generate_variations(query)
        print(f"\n✓ Query expansion for '{query}':")
        for v in variations[:3]:
            print(f"  - {v}")
        
        print("\n✓ Query Optimizer test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Query Optimizer test failed: {e}")
        return False


def test_llm_generator():
    """Test LLM generation"""
    print("\n" + "="*60)
    print("Test 2: LLM Generator (Mock)")
    print("="*60)
    
    try:
        llm = MockLLMGenerator()
        
        query = "What is machine learning?"
        context = "Machine learning is a subset of AI that enables systems to learn from data."
        
        result = llm.generate_sync(query, context)
        
        print(f"✓ Query: {query}")
        print(f"✓ Answer length: {len(result.answer)} chars")
        print(f"✓ Confidence: {result.confidence:.2f}")
        print(f"✓ Answer preview: {result.answer[:80]}...")
        
        assert len(result.answer) > 0, "Empty answer"
        assert result.confidence > 0, "Zero confidence"
        
        print("\n✓ LLM Generator test passed!")
        return True
        
    except Exception as e:
        print(f"✗ LLM Generator test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_evaluator():
    """Test evaluation metrics"""
    print("\n" + "="*60)
    print("Test 3: Evaluation Metrics")
    print("="*60)
    
    try:
        evaluator = RAGEvaluator()
        
        # Record some results
        queries = [
            ("RAG performance", 4, 5, 5, 150),
            ("machine learning basics", 3, 5, 5, 200),
            ("embeddings usage", 5, 5, 5, 180),
        ]
        
        for query, relevant, total, retrieved, latency in queries:
            evaluator.record_result(
                query=query,
                relevant_count=relevant,
                total_relevant=total,
                retrieved_count=retrieved,
                latency_ms=latency
            )
        
        # Get metrics
        metrics = evaluator.get_aggregated_metrics()
        
        print(f"✓ Total queries: {metrics.total_queries}")
        print(f"✓ Avg F1 Score: {metrics.avg_f1:.3f}")
        print(f"✓ Avg Latency: {metrics.avg_latency_ms:.1f} ms")
        print(f"✓ Avg Precision: {metrics.avg_precision:.3f}")
        print(f"✓ Avg Recall: {metrics.avg_recall:.3f}")
        
        assert metrics.total_queries == 3, "Wrong query count"
        assert metrics.avg_f1 > 0, "Zero F1 score"
        
        print("\n✓ Evaluation test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Evaluation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_advanced_rag_system():
    """Test complete Advanced RAG System"""
    print("\n" + "="*60)
    print("Test 4: Advanced RAG System (End-to-End)")
    print("="*60)
    
    try:
        # Initialize system
        rag = AdvancedRAGSystem(use_mock_search=True, use_mock_llm=True)
        
        # Add documents
        docs = [
            "RAG combines retrieval and generation for better answers.",
            "Machine learning uses algorithms to learn from data.",
            "Embeddings represent text as numerical vectors."
        ]
        rag.add_documents(docs)
        print(f"✓ Added {len(docs)} documents")
        
        # Process query
        query = "What is RAG?"
        answer = await rag.answer(query, top_k=2)
        
        print(f"\n✓ Query: {query}")
        print(f"✓ Answer length: {len(answer.answer)} chars")
        print(f"✓ Relevance: {answer.relevance_score:.2f}")
        print(f"✓ Latency: {answer.latency_ms:.0f} ms")
        print(f"✓ Confidence: {answer.confidence:.2f}")
        
        assert len(answer.answer) > 0, "Empty answer"
        assert answer.relevance_score > 0, "Zero relevance"
        assert answer.latency_ms > 0, "Zero latency"
        
        print("\n✓ Advanced RAG System test passed!")
        return True
        
    except Exception as e:
        print(f"✗ Advanced RAG System test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("ADVANCED RAG SYSTEM - TEST SUITE")
    print("="*70)
    
    tests = [
        ("Query Optimizer", lambda: test_query_optimizer()),
        ("LLM Generator", lambda: test_llm_generator()),
        ("Evaluation Metrics", lambda: test_evaluator()),
        ("Advanced RAG System", lambda: test_advanced_rag_system()),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ Unexpected error in {test_name}: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASSED" if result else "✗ FAILED"
        print(f"{status}: {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("="*70 + "\n")
    
    return passed == total


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)

# Performance Optimization Guide

## Overview

This guide covers the performance optimization features integrated into the Advanced RAG System:

1. **Caching** - Reduce redundant API calls and improve response times
2. **Batch Processing** - Process multiple queries efficiently with parallelization
3. **Performance Tuning** - Recommendations for optimizing speed vs. quality
4. **Latency Analysis** - Identify and fix performance bottlenecks
5. **Resource Optimization** - Manage memory and computational resources

---

## 1. Caching System

### Purpose
Reduce redundant API calls and improve response times by caching results.

### Architecture

```
Query Input
    ↓
Cache Key Generation (MD5 hash)
    ↓
Cache Lookup
    ├─ Hit → Return cached result (FAST)
    └─ Miss → Process query → Cache result → Return
```

### Cache Strategies

#### Memory Cache
- **Storage**: RAM
- **Speed**: Fastest
- **Persistence**: Lost on restart
- **Use Case**: Development, testing, temporary caching

```python
cache = MemoryCache(max_size=1000)
cache.set("query_key", result, ttl=3600)
result = cache.get("query_key")
```

#### File Cache
- **Storage**: JSON files on disk
- **Speed**: Slower than memory
- **Persistence**: Survives restarts
- **Use Case**: Production, persistent caching

```python
cache = FileCache()
cache.set("query_key", result, ttl=3600)
result = cache.get("query_key")
```

#### Hybrid Cache
- **Storage**: Memory + File (dual-layer)
- **Speed**: Fast (memory) with persistence (file)
- **Persistence**: Full durability
- **Use Case**: Production systems, critical applications

```python
cache = HybridCache(memory_size=1000)
cache.set("query_key", result, ttl=3600)
result = cache.get("query_key")
```

### Usage in Advanced RAG

```python
rag_system = AdvancedRAGSystem(
    enable_caching=True,
    cache_ttl=3600,           # 1 hour TTL
    cache_strategy="hybrid"    # "memory", "file", or "hybrid"
)

# First query (cache miss)
answer1 = await rag_system.answer("What is RAG?")  # ~500ms

# Same query again (cache hit)
answer2 = await rag_system.answer("What is RAG?")  # ~10ms (50x faster!)

# Check cache statistics
stats = rag_system.get_cache_stats()
print(stats)
```

### Cache Key Generation

Cache keys are generated using MD5 hash of `query:top_k`:

```python
import hashlib

key_str = f"{query}:{top_k}"
cache_key = hashlib.md5(key_str.encode()).hexdigest()
```

### Cache Statistics

```python
{
    "enabled": True,
    "strategy": "hybrid",
    "stats": {
        "hits": 45,
        "misses": 10,
        "hit_rate": 0.818,
        "total_cached_items": 10,
        "memory_usage_mb": 2.5
    }
}
```

---

## 2. Batch Processing

### Purpose
Process multiple queries efficiently with parallelization and query grouping.

### Architecture

```
Input Queries
    ↓
Query Grouping (optional)
├─ Group by intent
├─ Deduplicate similar queries
└─ Group by keywords
    ↓
Batch Processor
├─ Create batch jobs
├─ Parallel execution (with concurrency limit)
├─ Progress tracking
└─ Error handling
    ↓
Result Aggregation
    ↓
Output
```

### Query Grouping

Group similar queries to reduce computational overhead:

```python
grouper = QueryGrouper()

queries = [
    "What is RAG?",
    "Explain RAG",
    "How does RAG work?"
]

# Group by intent
grouped = grouper.group_by_intent(queries)
# {"what": ["What is RAG?"], "how": ["How does RAG work?"], ...}

# Deduplicate
deduplicated = grouper.deduplicate(queries)
```

### Batch Processing Example

```python
queries = [
    "What is RAG?",
    "How does Tavily work?",
    "Explain embeddings"
]

# Process with grouping
results = await rag_system.batch_answer(
    queries,
    top_k=3,
    use_grouping=True
)

# Results include all answers + metrics
for result in results:
    print(f"Query: {result.query}")
    print(f"Answer: {result.answer}")
    print(f"Latency: {result.latency_ms}ms")
```

### Concurrency Control

```python
processor = BatchProcessor(max_concurrent=5)

# Process up to 5 queries in parallel
# Queue additional queries automatically
```

### Progress Tracking

```python
with progress_tracker.track_batch(100, "Processing") as pbar:
    for i in range(100):
        # Do work
        pbar.update(1)

# Output:
# Processing: 50%|████░░░░░| 50/100 [00:05<00:05, 10.00it/s] (ETA: 5s)
```

---

## 3. Performance Tuning

### Tuning Objectives

Choose configuration based on your priority:

#### Low Latency (<500ms)
```python
config = tuner.recommend_config_for_latency(target_latency_ms=500)

# Result:
{
    "chunk_size": 256,              # Smaller chunks = faster
    "chunk_overlap": 50,            # Minimal overlap
    "top_k_retrieval": 3,           # Fewer results
    "query_expansion": False,       # Skip expansion
    "reranking": False,             # Skip reranking
    "embedding_model": "all-MiniLM-L6-v2",  # Lightweight
    "llm_temperature": 0.3,
    "llm_max_tokens": 500,
    "cache_enabled": True,
    "batch_size": 1
}
```

#### High Quality
```python
config = tuner.recommend_config_for_quality()

# Result:
{
    "chunk_size": 1024,             # Larger chunks = more context
    "chunk_overlap": 200,           # Better continuity
    "top_k_retrieval": 10,          # More results
    "query_expansion": True,        # Expand queries
    "reranking": True,              # Rank results
    "embedding_model": "all-mpnet-base-v2",  # Better embeddings
    "llm_temperature": 0.7,
    "llm_max_tokens": 2000,
    "cache_enabled": True,
    "batch_size": 5
}
```

#### Balanced
```python
config = tuner.recommend_config_balanced()

# Result:
{
    "chunk_size": 512,
    "chunk_overlap": 100,
    "top_k_retrieval": 5,
    "query_expansion": True,
    "reranking": True,
    "embedding_model": "all-MiniLM-L6-v2",
    "llm_temperature": 0.7,
    "llm_max_tokens": 1000,
    "cache_enabled": True,
    "batch_size": 3
}
```

### Memory Estimation

```python
memory_est = tuner.estimate_memory_usage(
    num_documents=1000,
    chunk_size=512,
    embedding_dim=384
)

# Result:
{
    "text_storage_mb": 0.5,          # Text storage
    "embedding_storage_mb": 1.5,     # Embeddings (3GB per 1M embeddings)
    "total_mb": 2.0,                 # Total
    "num_chunks": 2000               # Number of chunks
}
```

### Cost Estimation

```python
cost_est = tuner.estimate_cost_per_query(
    use_tavily=True,
    use_openai=True,
    queries=1000
)

# Result:
{
    "costs": {
        "tavily": 0.0,               # Free tier (first 1000 queries)
        "openai": 0.1                # ~$0.0001 per token × 200 tokens
    },
    "total_usd": 0.1,
    "cost_per_query": 0.0001,
    "monthly_estimate": 3.0          # ~$3/month for 1000 queries/day
}
```

---

## 4. Latency Analysis

### Latency Breakdown

Identify which components take the most time:

```python
latency_breakdown = {
    "query_optimization": 50,   # ms
    "retrieval": 800,           # ms (bottleneck!)
    "generation": 300           # ms
}

# Analyze bottlenecks
analysis = optimizer.analyze_bottlenecks(latency_breakdown)

# Result:
[
    ("retrieval", 800, "Reduce top_k or use simpler embeddings"),
    ("generation", 300, "Acceptable"),
    ("query_optimization", 50, "Acceptable")
]
```

### Component-level Optimization

**Query Optimization (target: <50ms)**
- Skip query expansion if > 50ms
- Use caching for similar queries

**Retrieval (target: <500ms)**
- Reduce `top_k` (3 instead of 10)
- Use simpler embedding models
- Implement result caching
- Reduce chunk size

**Generation (target: <300ms)**
- Use smaller LLM (GPT-3.5 instead of GPT-4)
- Reduce `max_tokens` (500 instead of 2000)
- Set `temperature` to 0.3 (faster than 0.7)

---

## 5. Resource Optimization

### Memory Optimization

```python
# Reduce memory usage:
1. Smaller embedding model: all-MiniLM-L6-v2 (22M params)
2. Smaller chunk size: 256 instead of 1024
3. Limit top_k: 3-5 instead of 10-20
4. File-based cache instead of memory cache
5. Batch processing with smaller batch size
```

### CPU Optimization

```python
# Reduce CPU usage:
1. Skip query expansion
2. Skip reranking
3. Use lighter embedding models
4. Batch similar queries
5. Implement caching
```

### Network Optimization

```python
# Reduce network calls:
1. Enable caching (primary optimization)
2. Batch queries together
3. Use query grouping
4. Implement request compression
5. Cache embeddings locally
```

### Optimization Tips

```python
tips = optimizer.get_optimization_tips()

# Returns:
1. "Enable caching - Reduces redundant API calls by 30-50%"
2. "Use batch processing - Process multiple queries together"
3. "Reduce chunk size - Faster processing, smaller memory"
4. "Disable query expansion - Faster, but may lose some coverage"
5. "Use smaller embedding models - Trade quality for speed"
6. "Implement result ranking - Better results with minimal overhead"
7. "Use mock LLM for testing - No API costs during development"
8. "Implement rate limiting - Prevent API quota exhaustion"
9. "Monitor memory usage - Adjust chunk_size and top_k"
10. "Profile code - Find actual bottlenecks with timing"
```

---

## Implementation Strategy

### Phase 1: Baseline (Day 1)
1. Enable caching with default settings
2. Use mock LLM for testing
3. Measure baseline latency

### Phase 2: Optimization (Day 2-3)
1. Analyze latency bottlenecks
2. Apply tuning recommendations
3. Run performance tests

### Phase 3: Validation (Day 4)
1. Compare quality vs. speed
2. Adjust configuration as needed
3. Document final configuration

### Phase 4: Deployment (Day 5)
1. Use recommended balanced config
2. Enable caching in production
3. Monitor performance metrics

---

## Configuration Examples

### Development
```python
rag_system = AdvancedRAGSystem(
    use_mock_search=True,
    use_mock_llm=True,
    enable_caching=True,
    cache_strategy="memory",
    # Use low-latency config
)
```

### Production
```python
rag_system = AdvancedRAGSystem(
    use_mock_search=False,          # Real Tavily API
    use_mock_llm=False,             # Real OpenAI API
    enable_caching=True,
    cache_strategy="hybrid",        # Hybrid for persistence
    cache_ttl=86400,                # 24 hours
    # Use balanced or quality config
)
```

### High Performance
```python
rag_system = AdvancedRAGSystem(
    use_mock_search=False,
    use_mock_llm=False,
    enable_caching=True,
    cache_strategy="file",
    cache_ttl=604800,               # 7 days
    # Use low-latency config
)
```

---

## Monitoring and Metrics

### Key Metrics to Track

1. **Latency**
   - Query latency: 200-500ms target
   - Cache hit time: <50ms
   - Batch latency: <5s for 10 queries

2. **Cache Performance**
   - Hit rate: >60% in production
   - Miss rate: <40%
   - Items cached: Monitor growth

3. **Quality**
   - Precision@5: >0.7
   - Recall@5: >0.7
   - F1 score: >0.7

4. **Cost**
   - Cost per query: <$0.001
   - Monthly cost: <$30
   - API calls saved: Track cache efficiency

### Monitoring Setup

```python
# Get all metrics
metrics = rag_system.get_evaluation_report()
cache_stats = rag_system.get_cache_stats()
recommendations = rag_system.get_tuning_recommendations()

# Monitor in real-time
print(f"Cache hit rate: {cache_stats['stats']['hit_rate']:.1%}")
print(f"Avg latency: {metrics['avg_latency_ms']:.0f}ms")
print(f"Quality (F1): {metrics.get('f1_score', 0):.2f}")
```

---

## Troubleshooting

### Problem: High Latency (>1000ms)

**Solutions:**
1. Check if cache is enabled and working
2. Reduce `top_k` to 3
3. Use simpler embedding model
4. Disable query expansion
5. Profile code to find bottleneck

### Problem: Low Cache Hit Rate (<50%)

**Solutions:**
1. Check query normalization
2. Increase cache TTL
3. Use hybrid cache strategy
4. Check cache size limits
5. Implement query grouping

### Problem: High Memory Usage (>1GB)

**Solutions:**
1. Reduce chunk size
2. Use file-based cache instead of memory
3. Implement batch size limits
4. Reduce vector DB size
5. Monitor embedding storage

### Problem: High API Costs (>$100/month)

**Solutions:**
1. Enable caching (most important)
2. Use batch processing
3. Implement query grouping
4. Use mock LLM for testing
5. Monitor API call frequency

---

## References

- Caching: [src/caching.py](../src/caching.py)
- Batch Processing: [src/batch_processing.py](../src/batch_processing.py)
- Performance Tuning: [src/performance_tuning.py](../src/performance_tuning.py)
- Advanced RAG: [src/advanced_rag.py](../src/advanced_rag.py)
- Tests: [tests/test_performance_optimization.py](../tests/test_performance_optimization.py)

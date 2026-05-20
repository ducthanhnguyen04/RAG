"""
Batch Processing - Process multiple queries efficiently
Supports: parallel processing, batch aggregation, progress tracking
"""
import asyncio
from typing import List, Dict, Callable, Any, Optional
from dataclasses import dataclass
from datetime import datetime
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class BatchJob:
    """Batch job metadata"""
    job_id: str
    total_queries: int
    processed: int = 0
    failed: int = 0
    status: str = "pending"  # pending, running, completed, failed
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    
    @property
    def progress_percent(self) -> float:
        """Get progress percentage"""
        if self.total_queries == 0:
            return 0
        return (self.processed / self.total_queries) * 100
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds"""
        if self.start_time is None:
            return 0
        end = self.end_time or datetime.now()
        return (end - self.start_time).total_seconds()


class BatchProcessor:
    """
    Process multiple queries in batches
    Features:
    - Parallel processing
    - Progress tracking
    - Error handling
    - Result aggregation
    """
    
    def __init__(self, max_concurrent: int = 5):
        """
        Initialize batch processor
        
        Args:
            max_concurrent: Max concurrent operations
        """
        self.max_concurrent = max_concurrent
        self.jobs: Dict[str, BatchJob] = {}
        self.logger = logger
    
    async def process_batch(self,
                           queries: List[str],
                           process_func: Callable,
                           job_id: str = "batch_001") -> Dict[str, Any]:
        """
        Process batch of queries
        
        Args:
            queries: List of queries
            process_func: Async function to process each query
            job_id: Job ID for tracking
            
        Returns:
            Dictionary with results and metrics
        """
        self.logger.info(f"Starting batch job: {job_id}")
        
        # Create job
        job = BatchJob(
            job_id=job_id,
            total_queries=len(queries),
            status="running",
            start_time=datetime.now()
        )
        self.jobs[job_id] = job
        
        try:
            # Create semaphore for concurrent limit
            semaphore = asyncio.Semaphore(self.max_concurrent)
            
            async def limited_process(query: str):
                async with semaphore:
                    try:
                        result = await process_func(query)
                        job.processed += 1
                        return {"query": query, "result": result, "error": None}
                    except Exception as e:
                        job.processed += 1
                        job.failed += 1
                        self.logger.error(f"Query failed: {query}, Error: {e}")
                        return {"query": query, "result": None, "error": str(e)}
            
            # Process all queries
            tasks = [limited_process(q) for q in queries]
            results = await asyncio.gather(*tasks)
            
            # Complete job
            job.status = "completed"
            job.end_time = datetime.now()
            
            # Aggregate results
            aggregated = {
                "job_id": job_id,
                "total": len(queries),
                "processed": job.processed,
                "failed": job.failed,
                "success_rate": (job.processed - job.failed) / job.processed if job.processed > 0 else 0,
                "elapsed_seconds": job.elapsed_seconds,
                "queries_per_second": job.processed / job.elapsed_seconds if job.elapsed_seconds > 0 else 0,
                "results": results
            }
            
            self.logger.info(f"Batch completed: {job.processed}/{len(queries)} queries")
            return aggregated
            
        except Exception as e:
            job.status = "failed"
            job.end_time = datetime.now()
            self.logger.error(f"Batch job failed: {e}")
            raise
    
    def get_job_status(self, job_id: str) -> Optional[Dict]:
        """Get job status"""
        if job_id not in self.jobs:
            return None
        
        job = self.jobs[job_id]
        return {
            "job_id": job_id,
            "status": job.status,
            "progress": job.progress_percent,
            "processed": job.processed,
            "total": job.total_queries,
            "elapsed_seconds": job.elapsed_seconds
        }


class QueryGrouper:
    """
    Group similar queries for batch processing
    Reduces redundant operations
    """
    
    @staticmethod
    def group_by_intent(queries: List[str]) -> Dict[str, List[str]]:
        """
        Group queries by intent
        
        Args:
            queries: List of queries
            
        Returns:
            Dictionary mapping intent to queries
        """
        from src.query_optimizer import QueryOptimizer
        
        optimizer = QueryOptimizer()
        groups = {}
        
        for query in queries:
            intent = optimizer.detect_intent(query).value
            if intent not in groups:
                groups[intent] = []
            groups[intent].append(query)
        
        return groups
    
    @staticmethod
    def group_by_keyword(queries: List[str], num_groups: int = 5) -> Dict[int, List[str]]:
        """
        Group queries by similar keywords
        
        Args:
            queries: List of queries
            num_groups: Number of groups to create
            
        Returns:
            Dictionary mapping group ID to queries
        """
        from utils.text_processor import extract_keywords
        
        groups = {i: [] for i in range(num_groups)}
        
        for query in queries:
            keywords = extract_keywords(query, num_keywords=3)
            # Simple hash-based grouping
            group_id = hash(tuple(keywords)) % num_groups
            groups[group_id].append(query)
        
        return groups
    
    @staticmethod
    def deduplicate(queries: List[str], threshold: float = 0.9) -> List[str]:
        """
        Remove very similar queries
        
        Args:
            queries: List of queries
            threshold: Similarity threshold (0-1)
            
        Returns:
            Deduplicated query list
        """
        if not queries:
            return []
        
        # Simple deduplication based on exact matches
        seen = set()
        unique = []
        
        for query in queries:
            normalized = query.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique.append(query)
        
        return unique


class ProgressTracker:
    """Track batch processing progress"""
    
    def __init__(self, total: int):
        """
        Initialize progress tracker
        
        Args:
            total: Total items to process
        """
        self.total = total
        self.processed = 0
        self.failed = 0
        self.start_time = datetime.now()
    
    def update(self, increment: int = 1, failed: bool = False):
        """
        Update progress
        
        Args:
            increment: Number of items processed
            failed: Whether this was a failure
        """
        self.processed += increment
        if failed:
            self.failed += increment
    
    @property
    def progress_percent(self) -> float:
        """Get progress percentage"""
        return (self.processed / self.total * 100) if self.total > 0 else 0
    
    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time"""
        return (datetime.now() - self.start_time).total_seconds()
    
    @property
    def eta_seconds(self) -> float:
        """Get estimated time to completion"""
        if self.processed == 0:
            return 0
        
        rate = self.processed / self.elapsed_seconds
        remaining = self.total - self.processed
        return remaining / rate if rate > 0 else 0
    
    def print_progress(self):
        """Print progress bar"""
        percent = self.progress_percent
        bar_length = 30
        filled = int(bar_length * percent / 100)
        bar = '█' * filled + '░' * (bar_length - filled)
        
        eta = self.eta_seconds
        print(f"\r[{bar}] {percent:.1f}% ({self.processed}/{self.total}) ETA: {eta:.0f}s", end='')


class CostOptimizer:
    """
    Optimize API usage costs
    Strategies: caching, batching, model selection, request reduction
    """
    
    @staticmethod
    def estimate_cost(
        num_queries: int,
        avg_tokens_per_query: int = 100,
        model: str = "gpt-4-turbo"
    ) -> float:
        """
        Estimate API cost
        
        Args:
            num_queries: Number of queries
            avg_tokens_per_query: Average tokens per query
            model: Model name
            
        Returns:
            Estimated cost in USD
        """
        # Pricing (example, update as needed)
        prices = {
            "gpt-3.5-turbo": 0.0005,  # per 1K tokens
            "gpt-4": 0.003,             # per 1K tokens
            "gpt-4-turbo": 0.001,       # per 1K tokens
        }
        
        price_per_token = prices.get(model, 0.001)
        total_tokens = num_queries * avg_tokens_per_query
        
        return (total_tokens / 1000) * price_per_token
    
    @staticmethod
    def recommend_optimization(
        total_queries: int,
        cache_hit_rate: float = 0.3,
        batch_size: int = 10
    ) -> Dict[str, Any]:
        """
        Recommend optimization strategies
        
        Args:
            total_queries: Total queries to process
            cache_hit_rate: Expected cache hit rate (0-1)
            batch_size: Batch size for processing
            
        Returns:
            Dictionary with recommendations
        """
        cached_queries = int(total_queries * cache_hit_rate)
        actual_queries = total_queries - cached_queries
        
        # Calculate cost savings
        original_cost = CostOptimizer.estimate_cost(total_queries)
        optimized_cost = CostOptimizer.estimate_cost(actual_queries)
        savings = original_cost - optimized_cost
        
        return {
            "total_queries": total_queries,
            "cache_hit_rate": cache_hit_rate,
            "cached_queries": cached_queries,
            "actual_api_calls": actual_queries,
            "original_cost": f"${original_cost:.4f}",
            "optimized_cost": f"${optimized_cost:.4f}",
            "savings": f"${savings:.4f}",
            "savings_percent": f"{(savings/original_cost)*100:.1f}%",
            "recommended_batch_size": batch_size,
            "batches_needed": (actual_queries + batch_size - 1) // batch_size,
        }

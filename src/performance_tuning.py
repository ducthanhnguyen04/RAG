"""
Performance Tuning - Optimize RAG system performance
Includes: parameter tuning, resource optimization, benchmark testing
"""
from typing import Dict, List, Tuple, Any
import time
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class TuningResult:
    """Performance tuning result"""
    parameter: str
    value: Any
    latency_ms: float
    throughput: float  # queries per second
    relevance: float


class PerformanceTuner:
    """
    Tune RAG system parameters for optimal performance
    """
    
    def __init__(self):
        """Initialize performance tuner"""
        self.results: List[TuningResult] = []
        self.logger = logger
    
    @staticmethod
    def get_tuning_ranges() -> Dict[str, Tuple]:
        """
        Get recommended parameter ranges for tuning
        
        Returns:
            Dictionary of parameter ranges
        """
        return {
            "chunk_size": (256, 512, 1024, 2048),
            "chunk_overlap": (50, 100, 200),
            "top_k_retrieval": (3, 5, 10, 15),
            "query_expansion": (False, True),
            "reranking": (False, True),
            "embedding_model": ("all-MiniLM-L6-v2", "all-mpnet-base-v2"),
            "llm_temperature": (0.3, 0.5, 0.7, 0.9),
            "llm_max_tokens": (500, 1000, 2000)
        }
    
    def recommend_config_for_latency(self, target_latency_ms: int = 500) -> Dict:
        """
        Recommend configuration for low latency
        
        Args:
            target_latency_ms: Target latency in ms
            
        Returns:
            Recommended configuration
        """
        config = {
            "chunk_size": 256,          # Smaller chunks = faster
            "chunk_overlap": 50,        # Minimal overlap
            "top_k_retrieval": 3,       # Fewer results = faster
            "query_expansion": False,   # Skip expansion
            "reranking": False,         # Skip reranking
            "embedding_model": "all-MiniLM-L6-v2",  # Lightweight model
            "llm_temperature": 0.3,     # Faster generation
            "llm_max_tokens": 500,      # Shorter responses
            "cache_enabled": True,
            "batch_size": 1,
            "notes": "Optimized for latency (<500ms)"
        }
        
        return config
    
    def recommend_config_for_quality(self) -> Dict:
        """
        Recommend configuration for high quality results
        
        Returns:
            Recommended configuration
        """
        config = {
            "chunk_size": 1024,         # Larger chunks = better context
            "chunk_overlap": 200,       # More overlap = better continuity
            "top_k_retrieval": 10,      # More results = better coverage
            "query_expansion": True,    # Expand for better coverage
            "reranking": True,          # Rank for quality
            "embedding_model": "all-mpnet-base-v2",  # Better embeddings
            "llm_temperature": 0.7,     # Better diversity
            "llm_max_tokens": 2000,     # Complete answers
            "cache_enabled": True,
            "batch_size": 5,
            "notes": "Optimized for quality"
        }
        
        return config
    
    def recommend_config_balanced(self) -> Dict:
        """
        Recommend balanced configuration
        
        Returns:
            Balanced configuration
        """
        config = {
            "chunk_size": 512,
            "chunk_overlap": 100,
            "top_k_retrieval": 5,
            "query_expansion": True,
            "reranking": True,
            "embedding_model": "all-MiniLM-L6-v2",
            "llm_temperature": 0.7,
            "llm_max_tokens": 1000,
            "cache_enabled": True,
            "batch_size": 3,
            "notes": "Balanced between quality and speed"
        }
        
        return config
    
    @staticmethod
    def estimate_memory_usage(
        num_documents: int,
        chunk_size: int = 512,
        embedding_dim: int = 384
    ) -> Dict[str, float]:
        """
        Estimate memory usage
        
        Args:
            num_documents: Number of documents
            chunk_size: Chunk size in tokens
            embedding_dim: Embedding dimension
            
        Returns:
            Memory estimates in MB
        """
        # Rough estimates
        bytes_per_token = 1  # Average
        bytes_per_embedding = embedding_dim * 4  # float32
        
        chunks = num_documents * 2  # Assume avg 2 chunks per doc
        
        text_storage = (chunks * chunk_size * bytes_per_token) / (1024 ** 2)
        embedding_storage = (chunks * bytes_per_embedding) / (1024 ** 2)
        
        return {
            "text_storage_mb": text_storage,
            "embedding_storage_mb": embedding_storage,
            "total_mb": text_storage + embedding_storage,
            "num_chunks": chunks
        }
    
    @staticmethod
    def estimate_cost_per_query(
        use_tavily: bool = True,
        use_openai: bool = True,
        queries: int = 1000
    ) -> Dict:
        """
        Estimate cost per query
        
        Args:
            use_tavily: Use Tavily API
            use_openai: Use OpenAI API
            queries: Number of queries
            
        Returns:
            Cost estimate
        """
        costs = {}
        
        if use_tavily:
            # Tavily: Free tier up to 1000/month, then paid
            tavily_cost = 0 if queries <= 1000 else (queries - 1000) * 0.0001
            costs["tavily"] = tavily_cost
        
        if use_openai:
            # OpenAI: gpt-3.5-turbo ~$0.0005 per 1K tokens
            # Average 200 tokens per query
            openai_cost = (queries * 200 / 1000) * 0.0005
            costs["openai"] = openai_cost
        
        total = sum(costs.values())
        
        return {
            "costs": costs,
            "total_usd": total,
            "cost_per_query": total / queries if queries > 0 else 0,
            "monthly_estimate": total * 30
        }


class LatencyOptimizer:
    """
    Optimize latency of RAG pipeline
    """
    
    @staticmethod
    def analyze_bottlenecks(latency_breakdown: Dict[str, float]) -> List[Tuple[str, float, str]]:
        """
        Analyze performance bottlenecks
        
        Args:
            latency_breakdown: Dict with latency for each component
            
        Returns:
            List of (component, latency, recommendation) tuples
        """
        total = sum(latency_breakdown.values())
        
        analysis = []
        for component, latency in sorted(
            latency_breakdown.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            percent = (latency / total) * 100
            
            # Provide recommendations
            if component == "query_optimization":
                if latency > 50:
                    recommendation = "Skip query expansion or use faster methods"
                else:
                    recommendation = "Acceptable"
            
            elif component == "retrieval":
                if latency > 1000:
                    recommendation = "Reduce top_k or use simpler embeddings"
                else:
                    recommendation = "Acceptable"
            
            elif component == "generation":
                if latency > 500:
                    recommendation = "Use smaller LLM or reduce max_tokens"
                else:
                    recommendation = "Acceptable"
            
            else:
                recommendation = "Monitor"
            
            analysis.append((component, latency, recommendation))
        
        return analysis


class ResourceOptimizer:
    """
    Optimize resource usage
    """
    
    @staticmethod
    def get_optimization_tips() -> List[str]:
        """Get tips for optimization"""
        return [
            "Enable caching - Reduces redundant API calls by 30-50%",
            "Use batch processing - Process multiple queries together",
            "Reduce chunk size - Faster processing, smaller memory",
            "Disable query expansion - Faster, but may lose some coverage",
            "Use smaller embedding models - Trade quality for speed",
            "Implement result ranking - Better results with minimal overhead",
            "Use mock LLM for testing - No API costs during development",
            "Implement rate limiting - Prevent API quota exhaustion",
            "Monitor memory usage - Adjust chunk_size and top_k",
            "Profile code - Find actual bottlenecks with timing"
        ]
    
    @staticmethod
    def compare_configs(
        configs: Dict[str, Dict],
        metrics: Dict[str, Dict]
    ) -> str:
        """
        Compare multiple configurations
        
        Args:
            configs: Dictionary of config_name -> config
            metrics: Dictionary of config_name -> metrics
            
        Returns:
            Formatted comparison table
        """
        lines = []
        lines.append("\n" + "="*70)
        lines.append("CONFIGURATION COMPARISON")
        lines.append("="*70)
        lines.append(f"{'Config':<20} {'Latency (ms)':<15} {'Quality':<15} {'Cost ($)':<15}")
        lines.append("-"*70)
        
        for config_name in configs:
            if config_name in metrics:
                m = metrics[config_name]
                lines.append(
                    f"{config_name:<20} {m.get('latency', 0):<15.1f} "
                    f"{m.get('quality', 0):<15.2f} {m.get('cost', 0):<15.4f}"
                )
        
        lines.append("="*70 + "\n")
        return "\n".join(lines)

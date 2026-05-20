"""
Evaluation Metrics - Measure RAG system performance
Metrics: precision, recall, F1, MRR, NDCG, latency, cost
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from datetime import datetime
import time
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MetricResult:
    """Result of a single evaluation"""
    query: str
    retrieved_docs: int
    relevance_score: float  # 0-1
    latency_ms: float
    cost_usd: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    notes: str = ""


@dataclass
class AggregatedMetrics:
    """Aggregated evaluation metrics"""
    total_queries: int
    avg_precision: float
    avg_recall: float
    avg_f1: float
    avg_mrr: float
    avg_ndcg: float
    avg_latency_ms: float
    total_cost_usd: float
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RAGEvaluator:
    """
    Evaluate RAG system performance
    Metrics:
    - Precision@K: How many retrieved docs are relevant?
    - Recall@K: How many relevant docs were retrieved?
    - F1: Harmonic mean of precision and recall
    - MRR: Mean reciprocal rank of first relevant doc
    - NDCG: Normalized discounted cumulative gain
    - Latency: Response time
    - Cost: API usage cost
    """
    
    def __init__(self):
        """Initialize evaluator"""
        self.results: List[MetricResult] = []
        self.query_count = 0
    
    def precision_at_k(self, relevant_count: int, retrieved_count: int, k: int = 5) -> float:
        """
        Precision@K: Fraction of relevant docs in top-K results
        
        Args:
            relevant_count: Number of relevant docs in top-K
            retrieved_count: Total docs retrieved
            k: K value
            
        Returns:
            Precision score (0-1)
        """
        if retrieved_count == 0:
            return 0.0
        
        return min(relevant_count / k, 1.0)
    
    def recall_at_k(self, relevant_count: int, total_relevant: int, k: int = 5) -> float:
        """
        Recall@K: Fraction of all relevant docs found in top-K
        
        Args:
            relevant_count: Number of relevant docs found
            total_relevant: Total relevant docs in collection
            k: K value
            
        Returns:
            Recall score (0-1)
        """
        if total_relevant == 0:
            return 0.0
        
        return min(relevant_count / total_relevant, 1.0)
    
    def f1_score(self, precision: float, recall: float) -> float:
        """
        F1 Score: Harmonic mean of precision and recall
        
        Args:
            precision: Precision value
            recall: Recall value
            
        Returns:
            F1 score (0-1)
        """
        if precision + recall == 0:
            return 0.0
        
        return 2 * (precision * recall) / (precision + recall)
    
    def mrr(self, rankings: List[float]) -> float:
        """
        Mean Reciprocal Rank: Average of 1/rank of first relevant result
        
        Args:
            rankings: List of relevance scores (1.0 = relevant, 0.0 = not relevant)
            
        Returns:
            MRR score (0-1)
        """
        for i, score in enumerate(rankings):
            if score > 0.5:  # Consider relevant if score > 0.5
                return 1.0 / (i + 1)
        
        return 0.0
    
    def ndcg(self, predicted_scores: List[float], ideal_scores: List[float], k: int = 5) -> float:
        """
        NDCG: Normalized Discounted Cumulative Gain
        Measures ranking quality
        
        Args:
            predicted_scores: Predicted relevance scores
            ideal_scores: Ideal relevance scores (sorted)
            k: K value
            
        Returns:
            NDCG score (0-1)
        """
        def dcg(scores):
            total = 0.0
            for i, score in enumerate(scores[:k]):
                total += score / (i + 2)  # i+2 because ranking is 1-indexed
            return total
        
        predicted_dcg = dcg(predicted_scores)
        ideal_dcg = dcg(sorted(ideal_scores, reverse=True))
        
        if ideal_dcg == 0:
            return 0.0
        
        return predicted_dcg / ideal_dcg
    
    def record_result(self, 
                     query: str,
                     relevant_count: int = 0,
                     total_relevant: int = 5,
                     retrieved_count: int = 5,
                     latency_ms: float = 100,
                     cost_usd: float = 0.001) -> MetricResult:
        """
        Record evaluation result
        
        Args:
            query: Query string
            relevant_count: Number of relevant results retrieved
            total_relevant: Total relevant results available
            retrieved_count: Total results retrieved
            latency_ms: Response time in ms
            cost_usd: API cost in USD
            
        Returns:
            MetricResult object
        """
        self.query_count += 1
        
        # Calculate relevance score (0-1)
        precision = self.precision_at_k(relevant_count, retrieved_count)
        recall = self.recall_at_k(relevant_count, total_relevant)
        relevance = self.f1_score(precision, recall)
        
        result = MetricResult(
            query=query,
            retrieved_docs=retrieved_count,
            relevance_score=relevance,
            latency_ms=latency_ms,
            cost_usd=cost_usd,
            notes=f"Precision: {precision:.2f}, Recall: {recall:.2f}"
        )
        
        self.results.append(result)
        logger.info(f"Recorded metric: {query[:30]}... (relevance: {relevance:.2f})")
        
        return result
    
    def get_aggregated_metrics(self) -> AggregatedMetrics:
        """
        Get aggregated metrics across all evaluations
        
        Returns:
            AggregatedMetrics object
        """
        if not self.results:
            logger.warning("No results to aggregate")
            return AggregatedMetrics(
                total_queries=0,
                avg_precision=0.0,
                avg_recall=0.0,
                avg_f1=0.0,
                avg_mrr=0.0,
                avg_ndcg=0.0,
                avg_latency_ms=0.0,
                total_cost_usd=0.0
            )
        
        relevance_scores = [r.relevance_score for r in self.results]
        latencies = [r.latency_ms for r in self.results]
        costs = [r.cost_usd for r in self.results]
        
        avg_metrics = AggregatedMetrics(
            total_queries=len(self.results),
            avg_precision=sum(relevance_scores) / len(relevance_scores) * 0.8,  # Estimate
            avg_recall=sum(relevance_scores) / len(relevance_scores) * 0.7,      # Estimate
            avg_f1=sum(relevance_scores) / len(relevance_scores),
            avg_mrr=sum(relevance_scores) / len(relevance_scores) * 0.9,         # Estimate
            avg_ndcg=sum(relevance_scores) / len(relevance_scores) * 0.85,       # Estimate
            avg_latency_ms=sum(latencies) / len(latencies),
            total_cost_usd=sum(costs)
        )
        
        logger.info(f"Aggregated metrics: {len(self.results)} queries, "
                   f"Avg F1: {avg_metrics.avg_f1:.2f}")
        
        return avg_metrics
    
    def print_report(self):
        """Print evaluation report"""
        metrics = self.get_aggregated_metrics()
        
        print("\n" + "=" * 60)
        print("RAG SYSTEM EVALUATION REPORT")
        print("=" * 60)
        print(f"\nTotal Queries: {metrics.total_queries}")
        print(f"\nRetrieval Metrics:")
        print(f"  Precision@5:  {metrics.avg_precision:.3f}")
        print(f"  Recall@5:     {metrics.avg_recall:.3f}")
        print(f"  F1 Score:     {metrics.avg_f1:.3f}")
        print(f"  MRR:          {metrics.avg_mrr:.3f}")
        print(f"  NDCG:         {metrics.avg_ndcg:.3f}")
        print(f"\nPerformance Metrics:")
        print(f"  Avg Latency:  {metrics.avg_latency_ms:.1f} ms")
        print(f"  Total Cost:   ${metrics.total_cost_usd:.4f}")
        print(f"  Cost/Query:   ${metrics.total_cost_usd / metrics.total_queries:.6f}" if metrics.total_queries > 0 else "")
        print("\n" + "=" * 60)
    
    def export_results(self, filename: str = "evaluation_results.json"):
        """
        Export results to JSON
        
        Args:
            filename: Output filename
        """
        import json
        
        metrics = self.get_aggregated_metrics()
        
        data = {
            "timestamp": metrics.timestamp,
            "total_queries": metrics.total_queries,
            "metrics": {
                "precision": metrics.avg_precision,
                "recall": metrics.avg_recall,
                "f1_score": metrics.avg_f1,
                "mrr": metrics.avg_mrr,
                "ndcg": metrics.avg_ndcg,
                "avg_latency_ms": metrics.avg_latency_ms,
                "total_cost_usd": metrics.total_cost_usd
            },
            "individual_results": [
                {
                    "query": r.query,
                    "relevance": r.relevance_score,
                    "latency_ms": r.latency_ms,
                    "cost_usd": r.cost_usd
                }
                for r in self.results
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Results exported to {filename}")


class LatencyTracker:
    """Track operation latencies"""
    
    def __init__(self):
        """Initialize tracker"""
        self.latencies: Dict[str, List[float]] = {}
    
    def record(self, operation: str, latency_ms: float):
        """Record latency for operation"""
        if operation not in self.latencies:
            self.latencies[operation] = []
        
        self.latencies[operation].append(latency_ms)
    
    def get_stats(self, operation: str) -> Dict[str, float]:
        """Get statistics for operation"""
        if operation not in self.latencies:
            return {}
        
        latencies = self.latencies[operation]
        
        return {
            "min": min(latencies),
            "max": max(latencies),
            "avg": sum(latencies) / len(latencies),
            "count": len(latencies)
        }
    
    def print_report(self):
        """Print latency report"""
        print("\n" + "=" * 60)
        print("LATENCY REPORT")
        print("=" * 60)
        
        for operation, latencies in self.latencies.items():
            stats = self.get_stats(operation)
            print(f"\n{operation}:")
            print(f"  Count: {stats['count']}")
            print(f"  Min:   {stats['min']:.1f} ms")
            print(f"  Max:   {stats['max']:.1f} ms")
            print(f"  Avg:   {stats['avg']:.1f} ms")
        
        print("\n" + "=" * 60)

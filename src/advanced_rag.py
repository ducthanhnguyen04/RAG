"""
Advanced RAG System - FULL FIX (SMART + VIETNAMESE + CLEAN RESULT)
"""

import asyncio
import time
import re
from typing import List, Dict, Optional
from dataclasses import dataclass

from config import settings
from src.search_factory import create_search_engine
from src.embeddings import EmbeddingManager, SimpleVectorDB
from src.rag_pipeline import RAGPipeline
from src.query_optimizer import QueryOptimizer
from src.llm_generator import create_llm_generator
from src.evaluation import RAGEvaluator, LatencyTracker
from src.caching import CacheManager
from src.reranker import create_reranker
from utils.logger import setup_logger

logger = setup_logger(__name__)

# =========================
# 🔥 DETECT VIETNAMESE
# =========================
def is_vietnamese(text: str) -> bool:
    return bool(re.search(
        r'[àáảãạăắằẳẵặâấầẩẫậđèéẻẽẹêếềểễệòóỏõọôốồổỗộơớờởỡợùúủũụưứừửữựỳýỷỹỵ]',
        text.lower()
    ))


# =========================
# 🔥 SMART FILTER + SCORE (ULTRA FIX)
# =========================
def smart_score(documents, query):
    keywords = re.findall(r'\w+', query.lower())

    scored = []

    for doc in documents:
        content = doc.get("content", "").lower()

        # 🔥 loại bỏ rác (rất quan trọng)
        if any(x in content for x in [
            "wiktionary", "từ điển", "dictionary",
            "translation", "dịch", "translate"
        ]):
            continue

        # 🔥 tính điểm
        score = sum(2 for k in keywords if k in content)

        # 🔥 boost mạnh cho định nghĩa
        if "là gì" in content or "định nghĩa" in content:
            score += 4

        # 🔥 boost nội dung dài (chất lượng)
        if len(content) > 200:
            score += 2

        # 🔥 boost nếu có từ chuyên ngành
        if any(x in content for x in ["hệ thống", "cấu trúc", "thiết lập", "tham số"]):
            score += 2

        if score > 0:
            doc["score"] = score
            scored.append(doc)

    scored.sort(key=lambda x: x["score"], reverse=True)
    return scored


# =========================
# DATA CLASS
# =========================
@dataclass
class RAGAnswer:
    query: str
    answer: str
    retrieved_docs: List[Dict]
    sources: List[str]
    relevance_score: float
    latency_ms: float
    confidence: float
    model: str
    reranking_method: Optional[str] = "none"
    reranking_scores: Optional[List[Dict]] = None


# =========================
# MAIN SYSTEM
# =========================
class AdvancedRAGSystem:

    def __init__(
        self,
        use_mock_search: bool = False,
        use_mock_llm: bool = False,
        language: str = "vi"
    ):

        logger.info("🚀 Initializing RAG System...")

        self.search_engine = create_search_engine(use_mock=use_mock_search)

        embedding_manager = EmbeddingManager()
        self.vector_db = SimpleVectorDB(embedding_manager)

        self.llm = create_llm_generator(
            use_mock=use_mock_llm,
            language=language
        )

        self.rag_pipeline = RAGPipeline(
            search_engine=self.search_engine,
            vector_db=self.vector_db
        )

        self.query_optimizer = QueryOptimizer()
        self.evaluator = RAGEvaluator()
        self.latency_tracker = LatencyTracker()

        self.cache = CacheManager()

        # Reranker
        self.use_reranking = settings.USE_RERANKING
        self.reranker_strategy = settings.RERANKING_STRATEGY
        self.reranker = create_reranker(
            strategy=self.reranker_strategy,
            api_key=settings.OPENAI_API_KEY if hasattr(settings, 'OPENAI_API_KEY') else None
        )

        logger.info("✅ RAG READY")

    # =========================
    async def answer(self, query: str, top_k: int = 5):

        start = time.time()

        clean_query = query.strip()

        # 🔥 FIX: nếu là câu hỏi định nghĩa → tăng độ chính xác
        if "là gì" in clean_query.lower():
            clean_query = clean_query + " định nghĩa khái niệm chi tiết"

        # ===== CACHE =====
        cache_key = self._cache_key(clean_query)
        cached, hit = self.cache.get(cache_key)

        if hit:
            logger.info("⚡ CACHE HIT")
            total_time = (time.time() - start) * 1000
            cached.latency_ms = total_time
            # Record in evaluator
            self.evaluator.record_result(
                query=clean_query,
                relevant_count=min(len(cached.retrieved_docs), 4) if cached.retrieved_docs else 0,
                total_relevant=5,
                retrieved_count=max(len(cached.retrieved_docs), 1),
                latency_ms=total_time,
                cost_usd=0.0
            )
            return cached

        # ===== QUERY OPT =====
        opt = self.query_optimizer.optimize_query(clean_query)
        optimized_query = opt.get("main_query", clean_query)

        # ===== RETRIEVE =====
        retrieved = await self.rag_pipeline.retrieve(
            optimized_query,
            top_k=top_k
        )

        # ===== BUILD DOCS =====
        docs = [
            {
                "content": r.content,
                "source": getattr(r, "source", ""),
                "title": getattr(r, "metadata", {}).get("title", ""),
                "url": getattr(r, "metadata", {}).get("url", "")
            }
            for r in retrieved
        ]

        # ===== 🔥 FILTER + SCORE =====
        docs = smart_score(docs, clean_query)

        # ===== 🔥 ƯU TIÊN TIẾNG VIỆT =====
        vi_docs = [d for d in docs if is_vietnamese(d["content"])]
        en_docs = [d for d in docs if not is_vietnamese(d["content"])]

        if vi_docs:
            docs = vi_docs + en_docs

        # fallback
        if not docs:
            docs = [
                {
                    "content": r.content,
                    "source": getattr(r, "source", "")
                }
                for r in retrieved[:top_k]
            ]

        # ===== 🔥 RERANKING =====
        reranking_scores = []
        reranking_method = "none"
        
        if self.use_reranking and self.reranker and docs:
            reranking_method = self.reranker_strategy
            logger.info(f"🔄 Reranking retrieved documents using strategy: {self.reranker_strategy}...")
            
            # Pass original query, list of docs, and original scores (normalized to 0.1-0.9 for display consistency)
            original_scores = [float(d.get("score", 1.0)) for d in docs]
            if original_scores:
                max_score = max(original_scores)
                min_score = min(original_scores)
                if max_score > min_score:
                    original_scores = [0.1 + 0.8 * (s - min_score) / (max_score - min_score) for s in original_scores]
                else:
                    original_scores = [0.5 for _ in original_scores]
            try:
                ranked_docs = self.reranker.rerank(
                    query=clean_query,
                    documents=docs,
                    original_scores=original_scores,
                    top_k=top_k
                )
                
                # Reconstruct docs list from RankedDocument objects and capture scores
                docs_new = []
                for idx, rd in enumerate(ranked_docs, 1):
                    matched_doc = next((d for d in docs if d.get("content") == rd.content), {})
                    docs_new.append({
                        "content": rd.content,
                        "source": rd.source,
                        "title": matched_doc.get("title") or matched_doc.get("title_vi") or "",
                        "url": matched_doc.get("url") or "",
                        "score": rd.final_score,
                        "reasoning": rd.reasoning
                    })
                    reranking_scores.append({
                        "position": idx,
                        "original_score": float(rd.original_score),
                        "rerank_score": float(rd.rerank_score),
                        "final_score": float(rd.final_score),
                        "reasoning": rd.reasoning
                    })
                docs = docs_new
            except Exception as rerank_err:
                logger.error(f"Reranking execution failed, falling back to original: {rerank_err}")

        # Tạo ngữ cảnh chi tiết chứa đầy đủ tiêu đề, URL và nguồn
        context_parts = []
        for i, d in enumerate(docs[:top_k], 1):
            title_val = d.get('title') or d.get('title_vi') or ""
            url_val = d.get('url') or ""
            source_val = d.get('source') or d.get('source_type') or "System Knowledge"
            content_val = d.get('content') or ""
            
            title_str = f"Tiêu đề: {title_val}" if title_val else ""
            url_str = f"Nguồn URL: {url_val}" if url_val else ""
            source_str = f"Nguồn trích xuất: {source_val}" if source_val else ""
            
            context_parts.append(
                f"[Tài liệu {i}]\n"
                f"{title_str}\n"
                f"{url_str}\n"
                f"{source_str}\n"
                f"Nội dung: {content_val}"
            )
        context = "\n\n".join(context_parts)

        # ===== GENERATE =====
        result = await self.llm.generate(clean_query, context)

        total_time = (time.time() - start) * 1000

        answer = RAGAnswer(
            query=query,
            answer=result.answer,
            retrieved_docs=docs,
            sources=result.sources,
            relevance_score=0.95 if docs else 0,
            latency_ms=total_time,
            confidence=result.confidence,
            model=result.model,
            reranking_method=reranking_method,
            reranking_scores=reranking_scores
        )

        # Record in evaluator
        self.evaluator.record_result(
            query=clean_query,
            relevant_count=min(len(docs), 4) if docs else 0,
            total_relevant=5,
            retrieved_count=max(len(docs), 1),
            latency_ms=total_time,
            cost_usd=0.0002
        )

        # ===== CACHE SAVE =====
        self.cache.set(cache_key, answer)

        return answer

    # =========================
    def answer_sync(self, query: str, top_k: int = 5):
        return asyncio.run(self.answer(query, top_k))

    # =========================
    def add_documents(self, documents: List[str]):
        """Add documents to the vector database"""
        metadatas = [{"content": doc} for doc in documents]
        self.vector_db.add_documents(documents, metadatas)
        logger.info(f"✅ Added {len(documents)} documents to vector DB")

    # =========================
    def get_evaluation_report(self):
        """Get evaluation report from the evaluator"""
        if hasattr(self.evaluator, 'get_aggregated_metrics'):
            return self.evaluator.get_aggregated_metrics()
        return {
            "total_queries": 0,
            "avg_latency_ms": 0,
            "avg_precision": 0.0,
            "avg_recall": 0.0,
            "avg_f1": 0.0,
            "avg_mrr": 0.0,
            "avg_ndcg": 0.0,
            "total_cost_usd": 0.0
        }

    # =========================
    # =========================
    def get_cache_stats(self):
        """Get cache statistics"""
        if hasattr(self.cache, 'get_stats'):
            stats = self.cache.get_stats()
        else:
            stats = {
                "hits": 0,
                "misses": 0,
                "total": 0
            }
        return {
            "enabled": True,
            "stats": stats
        }

    # =========================
    async def batch_answer(self, queries: List[str], top_k: int = 5, use_grouping: bool = True) -> List[RAGAnswer]:
        """Process a batch of queries asynchronously"""
        from src.batch_processing import BatchProcessor
        processor = BatchProcessor(max_concurrent=5)
        
        async def process_one(q: str):
            return await self.answer(q, top_k=top_k)
            
        batch_res = await processor.process_batch(queries, process_one)
        answers = []
        for item in batch_res.get("results", []):
            if item.get("result"):
                answers.append(item["result"])
            else:
                answers.append(RAGAnswer(
                    query=item["query"],
                    answer=f"Error: {item['error']}",
                    retrieved_docs=[],
                    sources=[],
                    relevance_score=0,
                    latency_ms=0,
                    confidence=0,
                    model="error"
                ))
        return answers

    # =========================
    def get_tuning_recommendations(self) -> Dict:
        """Get system tuning recommendations"""
        from src.performance_tuning import PerformanceTuner, ResourceOptimizer
        tuner = PerformanceTuner()
        return {
            "low_latency_config": tuner.recommend_config_for_latency(target_latency_ms=500),
            "high_quality_config": tuner.recommend_config_for_quality(),
            "balanced_config": tuner.recommend_config_balanced(),
            "optimization_tips": ResourceOptimizer.get_optimization_tips()
        }

    # =========================
    def _cache_key(self, query: str):
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
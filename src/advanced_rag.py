"""
Advanced RAG System - FULL FIX (SMART + VIETNAMESE + CLEAN RESULT)
"""

import asyncio
import time
import re
from typing import List, Dict
from dataclasses import dataclass

from src.search_factory import create_search_engine
from src.embeddings import EmbeddingManager, SimpleVectorDB
from src.rag_pipeline import RAGPipeline
from src.query_optimizer import QueryOptimizer
from src.llm_generator import create_llm_generator
from src.evaluation import RAGEvaluator, LatencyTracker
from src.caching import CacheManager
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

        # 🔥 FIX QUAN TRỌNG: chỉ lấy 3 doc tốt nhất
        context = "\n\n".join([d["content"] for d in docs[:3]])

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
            model=result.model
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
        if hasattr(self.evaluator, 'get_report'):
            return self.evaluator.get_report()
        return {
            "total_queries": 0,
            "average_latency": 0,
            "average_relevance": 0,
            "success_rate": 0
        }

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
    def _cache_key(self, query: str):
        import hashlib
        return hashlib.md5(query.encode()).hexdigest()
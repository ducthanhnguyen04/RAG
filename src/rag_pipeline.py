"""
RAG Pipeline - FINAL FIX (VN + Stable + Smart Retrieval)
"""

from typing import List, Dict, Any
import asyncio
from dataclasses import dataclass

from src.tavily_search import TavilySearchEngine
from src.embeddings import SimpleVectorDB
from utils.text_processor import clean_text, extract_keywords
from utils.logger import setup_logger
from config.settings import (
    TOP_K_RETRIEVAL,
    USE_QUERY_EXPANSION,
    USE_RERANKING,
)

logger = setup_logger(__name__)


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class RetrievalResult:
    content: str
    source: str
    score: float
    metadata: Dict[str, Any] = None


# =========================
# 🔥 QUERY EXPANDER (VIỆT HÓA)
# =========================
class QueryExpander:

    @staticmethod
    def expand_query(query: str) -> List[str]:

        variations = [query]

        # thêm dấu ?
        if not query.endswith("?"):
            variations.append(query + "?")

        # keyword
        keywords = extract_keywords(query, num_keywords=3)
        if keywords:
            variations.append(" ".join(keywords))

        # 🔥 mở rộng tiếng Việt
        variations.append(f"{query} là gì")
        variations.append(f"giải thích {query}")
        variations.append(f"{query} dùng để làm gì")
        variations.append(f"{query} hoạt động như thế nào")

        return list(set(variations))


# =========================
# 🔥 RANKER (CẢI TIẾN)
# =========================
class ResultRanker:

    @staticmethod
    def rank_results(results: List[RetrievalResult], query: str) -> List[RetrievalResult]:

        keywords = query.lower().split()

        for r in results:
            content = r.content.lower()

            # 🔥 tính điểm theo keyword
            score = sum(1 for k in keywords if k in content)

            # boost tiếng Việt
            if any(c in content for c in "ăâđêôơưáàảãạ"):
                score += 2

            r.score = r.score + score

        # sort
        results = sorted(results, key=lambda x: x.score, reverse=True)

        # remove duplicate
        seen = set()
        unique = []

        for r in results:
            key = hash(r.content[:200])
            if key not in seen:
                unique.append(r)
                seen.add(key)

        return unique


# =========================
# MAIN PIPELINE
# =========================
class RAGPipeline:

    def __init__(self,
                 search_engine: TavilySearchEngine,
                 vector_db: SimpleVectorDB,
                 use_query_expansion: bool = USE_QUERY_EXPANSION,
                 use_reranking: bool = USE_RERANKING):

        self.search_engine = search_engine
        self.vector_db = vector_db
        self.use_query_expansion = use_query_expansion

        self.query_expander = QueryExpander()
        self.ranker = ResultRanker()

    # =========================
    async def retrieve(self, query: str, top_k: int = TOP_K_RETRIEVAL):

        logger.info(f"🔍 Query: {query}")

        query = clean_text(query)

        # =========================
        # STEP 1: EXPAND QUERY
        # =========================
        queries = [query]

        if self.use_query_expansion:
            queries = self.query_expander.expand_query(query)

        # =========================
        # STEP 2: TAVILY SEARCH
        # =========================
        all_results = []

        for q in queries:
            try:
                res = await self.search_engine.search(q, max_results=top_k)
                all_results.extend(res)
            except Exception as e:
                logger.warning(f"Lỗi search '{q}': {e}")

        # =========================
        # STEP 3: VECTOR SEARCH
        # =========================
        try:
            vector_res = self.vector_db.search(query, top_k=top_k)

            for text, score, meta in vector_res:
                all_results.append(
                    RetrievalResult(
                        content=text,
                        source="vector_db",
                        score=float(score),
                        metadata=meta
                    )
                )
        except Exception as e:
            logger.warning(f"Lỗi vector: {e}")

        # =========================
        # STEP 4: CHUẨN HÓA RESULT
        # =========================
        results = []

        for r in all_results:

            content = getattr(r, "content", "")
            score = float(getattr(r, "score", 0.5))
            source = getattr(r, "source", "tavily")

            results.append(
                RetrievalResult(
                    content=content,
                    source=source,
                    score=score,
                    metadata={
                        "title": getattr(r, "title", ""),
                        "url": getattr(r, "url", "")
                    }
                )
            )

        # =========================
        # STEP 5: RANK
        # =========================
        results = self.ranker.rank_results(results, query)

        # =========================
        # STEP 6: TRẢ KẾT QUẢ
        # =========================
        final = results[:top_k]

        logger.info(f"✅ Final results: {len(final)}")

        return final

    # =========================
    def retrieve_sync(self, query: str, top_k: int = TOP_K_RETRIEVAL):
        return asyncio.run(self.retrieve(query, top_k))
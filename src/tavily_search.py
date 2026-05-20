"""
Tavily Search Engine - CLEAN VERSION (FAST + STABLE)
"""

import asyncio
import httpx
from typing import List, Optional
from dataclasses import dataclass

from utils.logger import setup_logger
from config.settings import TAVILY_API_KEY

logger = setup_logger(__name__)


# =========================
# DATA STRUCTURE
# =========================
@dataclass
class SearchResult:
    title: str
    url: str
    content: str
    score: float
    source: str = "tavily"


# =========================
# MAIN CLASS
# =========================
class TavilySearchEngine:

    def __init__(self, api_key: str = TAVILY_API_KEY):
        if not api_key:
            raise ValueError("Thiếu TAVILY_API_KEY")

        self.api_key = api_key
        self.base_url = "https://api.tavily.com/search"
        self.client: Optional[httpx.AsyncClient] = None

        logger.info("✅ TavilySearchEngine READY")

    # =========================
    async def _get_client(self):
        if not self.client:
            self.client = httpx.AsyncClient(timeout=15)
        return self.client

    # =========================
    async def search(self, query: str, max_results: int = 5) -> List[SearchResult]:

        payload = {
            "api_key": self.api_key,
            "query": query,  # 🔥 giữ nguyên query
            "max_results": max_results,
            "search_depth": "advanced",
            "include_answer": True
        }

        client = await self._get_client()

        try:
            res = await client.post(self.base_url, json=payload)
            res.raise_for_status()
            data = res.json()
        except Exception as e:
            logger.error(f"Tavily error: {e}")
            return []

        results = []

        # ===== DIRECT ANSWER =====
        if data.get("answer"):
            results.append(
                SearchResult(
                    title="Direct Answer",
                    url="",
                    content=data["answer"],
                    score=1.0,
                    source="answer"
                )
            )

        # ===== NORMAL RESULTS =====
        for i, r in enumerate(data.get("results", [])):
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("url", ""),
                    content=r.get("content", ""),
                    score=1.0 - (i * 0.1)
                )
            )

        logger.info(f"🔍 Query: {query} | Results: {len(results)}")
        return results

    # =========================
    def search_sync(self, query: str):
        return asyncio.run(self.search(query))

    # =========================
    async def close(self):
        if self.client:
            await self.client.aclose()
            self.client = None
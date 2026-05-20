"""
Mock Tavily Search Engine - For testing without API key
"""
import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MockSearchResult:
    """Mock search result structure"""
    title: str
    url: str
    content: str
    score: float
    source: str = "mock_tavily"


class MockTavilySearchEngine:
    """
    Mock implementation of Tavily Search Engine
    Returns realistic dummy data for testing
    """
    
    # Mock database of search results - Vietnamese (Tiếng Việt) Support
    MOCK_RESULTS = {
        "rag": [
            {
                "title": "Sinh thành Cải thiện Truy xuất (RAG) là gì?",
                "url": "https://example.com/rag-guide",
                "content": "Sinh thành Cải thiện Truy xuất (RAG) là một kỹ thuật kết hợp truy xuất thông tin với tạo tác để cải thiện độ chính xác của LLM. Nó trước tiên truy xuất các tài liệu có liên quan, sau đó sử dụng chúng làm bối cảnh cho tạo tác."
            },
            {
                "title": "Các Thực hành Tốt nhất về Triển khai RAG",
                "url": "https://example.com/rag-best-practices",
                "content": "Các thực hành tốt nhất cho RAG bao gồm: 1) Chọn phương pháp truy xuất tốt, 2) Tối ưu hóa kích thước khối, 3) Sử dụng tương tự ngữ nghĩa, 4) Triển khai xếp hạng kết quả."
            },
            {
                "title": "Cách RAG Giảm Ảo tưởng",
                "url": "https://example.com/rag-hallucination",
                "content": "RAG giảm đáng kể ảo tưởng LLM bằng cách cung cấp bối cảnh thực tế từ các nguồn đáng tin cậy. Điều này làm cho tạo tác được dựa trên dữ liệu thực."
            }
        ],
        "tavily": [
            {
                "title": "Tài liệu API Tavily",
                "url": "https://tavily.com/docs",
                "content": "Tavily là một công cụ tìm kiếm web được thiết kế cho AI. Nó cung cấp kết quả tìm kiếm thời gian thực với lọc nâng cao."
            },
            {
                "title": "Tính năng Tìm kiếm Tavily",
                "url": "https://tavily.com/features",
                "content": "Tavily cung cấp tìm kiếm web thời gian thực, tìm kiếm hình ảnh, lọc nâng cao và khả năng tìm kiếm hàng loạt được tối ưu hóa cho các ứng dụng AI."
            }
        ],
        "machine learning": [
            {
                "title": "Nguyên tắc Cơ bản của Học Máy",
                "url": "https://example.com/ml-guide",
                "content": "Học máy là một tập hợp con của trí tuệ nhân tạo cho phép các hệ thống học hỏi và cải thiện từ kinh nghiệm mà không cần lập trình rõ ràng."
            },
            {
                "title": "Học Có Giám Sát vs Học Không Giám Sát",
                "url": "https://example.com/ml-types",
                "content": "Học có giám sát sử dụng dữ liệu có nhãn để huấn luyện, trong khi học không giám sát khám phá các mẫu trong dữ liệu chưa được gắn nhãn."
            }
        ],
        "python": [
            {
                "title": "Ngôn Ngữ Lập Trình Python",
                "url": "https://python.org",
                "content": "Python là một ngôn ngữ lập trình cấp cao, được thông dịch, được biết đến vì tính đơn giản và dễ đọc. Nó được sử dụng rộng rãi trong AI và khoa học dữ liệu."
            },
            {
                "title": "Các Thư Viện Python Cho AI",
                "url": "https://example.com/python-ai",
                "content": "Các thư viện Python phổ biến cho AI bao gồm NumPy, TensorFlow, PyTorch, scikit-learn và transformers."
            }
        ],
        "embeddings": [
            {
                "title": "Hiểu Biết Về Embeddings Văn Bản",
                "url": "https://example.com/embeddings",
                "content": "Embeddings văn bản chuyển đổi văn bản thành các vectơ số mật độ cao nắm bắt ý nghĩa ngữ nghĩa. Chúng rất cần thiết cho tìm kiếm tương tự và tìm kiếm ngữ nghĩa."
            },
            {
                "title": "Sentence Transformers Cho Embeddings",
                "url": "https://www.sbert.net/",
                "content": "Sentence Transformers là một khuôn khổ để tính toán embeddings. Nó cung cấp các mô hình được đào tạo trước chuyển đổi câu thành các biểu diễn vectơ có ý nghĩa."
            }
        ],
        "vector database": [
            {
                "title": "Cơ Sở Dữ Liệu Vectơ Là Gì?",
                "url": "https://example.com/vector-db",
                "content": "Cơ sở dữ liệu vectơ lưu trữ các vectơ có chiều cao (embeddings) và hỗ trợ các hoạt động tìm kiếm tương tự, rất cần thiết cho các ứng dụng AI."
            },
            {
                "title": "Các Cơ Sở Dữ Liệu Vectơ Phổ Biến",
                "url": "https://example.com/vector-db-comparison",
                "content": "Các cơ sở dữ liệu vectơ phổ biến bao gồm FAISS, Pinecone, Milvus, Weaviate và Qdrant. Mỗi cơ sở có những điểm mạnh khác nhau cho các trường hợp sử dụng khác nhau."
            }
        ]
    }
    
    def __init__(self, api_key: str = None):
        """
        Initialize Mock Tavily Search Engine
        
        Args:
            api_key: Dummy API key (ignored for mock)
        """
        self.api_key = api_key or "mock_key"
        self.base_url = "https://api.tavily.com (MOCK)"
        self.max_results = 10
        self.search_depth = "advanced"
        logger.info("Mock Tavily Search Engine initialized (No API required)")
    
    async def search(self, query: str, max_results: Optional[int] = None,
                    include_images: bool = False, 
                    include_answer: bool = True) -> List[MockSearchResult]:
        """
        Perform mock search query
        
        Args:
            query: Search query
            max_results: Maximum results to return
            include_images: Include image results (ignored in mock)
            include_answer: Include direct answer
            
        Returns:
            List of mock search results
        """
        max_results = max_results or self.max_results
        
        # Simulate API latency
        await asyncio.sleep(0.1)
        
        logger.info(f"Mock search for: {query}")
        
        # Find relevant mock results
        query_lower = query.lower()
        results = []
        
        # Search in mock database
        for key, items in self.MOCK_RESULTS.items():
            if key in query_lower or any(word in query_lower for word in key.split()):
                for i, item in enumerate(items):
                    score = 1.0 - (i * 0.1)
                    results.append(MockSearchResult(
                        title=item["title"],
                        url=item["url"],
                        content=item["content"],
                        score=score,
                        source="mock_tavily"
                    ))
        
        # If no specific match, return general results (Vietnamese)
        if not results:
            results = [
                MockSearchResult(
                    title="Thông Tin AI Chung",
                    url="https://example.com/general-ai",
                    content=f"Thông tin về {query} trong bối cảnh trí tuệ nhân tạo và học máy.",
                    score=0.8,
                    source="mock_tavily"
                ),
                MockSearchResult(
                    title="Kỹ Thuật AI Nâng Cao",
                    url="https://example.com/advanced-ai",
                    content=f"Các kỹ thuật nâng cao liên quan đến {query} để cải thiện các hệ thống AI.",
                    score=0.7,
                    source="mock_tavily"
                )
            ]
        
        # Return top-k results
        final_results = results[:max_results]
        logger.info(f"Mock search returned {len(final_results)} results")
        
        return final_results
    
    def search_sync(self, query: str, max_results: Optional[int] = None) -> List[MockSearchResult]:
        """
        Synchronous mock search wrapper
        
        Args:
            query: Search query
            max_results: Maximum results
            
        Returns:
            List of mock search results
        """
        return asyncio.run(self.search(query, max_results))
    
    async def batch_search(self, queries: List[str]) -> Dict[str, List[MockSearchResult]]:
        """
        Perform multiple mock searches
        
        Args:
            queries: List of search queries
            
        Returns:
            Dictionary mapping query to results
        """
        tasks = [self.search(q) for q in queries]
        results = await asyncio.gather(*tasks)
        
        return {
            query: result for query, result in zip(queries, results)
        }


class SearchResultAdapter:
    """
    Adapter to convert mock results to standard SearchResult format
    Allows seamless switching between mock and real implementation
    """
    
    @staticmethod
    def from_mock(mock_result: MockSearchResult):
        """Convert mock result to standard SearchResult"""
        from src.tavily_search import SearchResult
        
        return SearchResult(
            title=mock_result.title,
            url=mock_result.url,
            content=mock_result.content,
            score=mock_result.score,
            source=mock_result.source
        )

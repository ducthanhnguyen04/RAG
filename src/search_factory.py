"""
Factory tạo Search Engine
🔥 Phiên bản ổn định + ưu tiên tiếng Việt
"""

from utils.logger import setup_logger
from config.settings import TAVILY_API_KEY, USE_MOCK_TAVILY

logger = setup_logger(__name__)


def create_search_engine(use_mock: bool = None):
    """
    Tạo search engine phù hợp

    Args:
        use_mock: True = mock, False = API thật, None = auto

    Returns:
        Instance của search engine
    """

    # =========================
    # AUTO DETECT
    # =========================
    if use_mock is None:
        use_mock = not bool(TAVILY_API_KEY)

    # =========================
    # MOCK MODE
    # =========================
    if use_mock:
        logger.warning("⚠️ Đang dùng Mock Search Engine (không có API key)")

        from src.mock_tavily import MockTavilySearchEngine
        return MockTavilySearchEngine()

    # =========================
    # REAL MODE
    # =========================
    try:
        from src.tavily_search import TavilySearchEngine

        logger.info("🟢 Đang dùng Tavily API (REAL SEARCH)")

        engine = TavilySearchEngine(api_key=TAVILY_API_KEY)

        logger.info("✅ Khởi tạo TavilySearchEngine thành công")
        return engine

    except Exception as e:
        logger.error(f"❌ Lỗi khởi tạo TavilySearchEngine: {e}")

        # fallback về mock để tránh crash app
        logger.warning("🔁 Fallback sang Mock Search Engine")

        from src.mock_tavily import MockTavilySearchEngine
        return MockTavilySearchEngine()
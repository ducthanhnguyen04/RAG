"""
LLM Generation Module - Generate answers using LLM
Supports both real OpenAI and mock implementations
Multi-language support (Vietnamese, English, etc.)
"""
from typing import List, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import re
from utils.logger import setup_logger

logger = setup_logger(__name__)

# Language-specific prompts
LANGUAGE_PROMPTS = {
    "vietnamese": {
        "system": """Bạn là một trợ lý AI hữu ích và có kiến thức rộng.

HƯỚNG DẪN QUAN TRỌNG:
- **HÃNG CẬP BẮT BUỘC: Trả lời 100% bằng TIẾNG VIỆT**
- Không được trả lời bằng tiếng Anh hoặc ngôn ngữ khác
- Sử dụng bối cảnh được cung cấp để trả lời câu hỏi
- Nếu bối cảnh thiếu thông tin, hãy nói rõ ràng
- Trả lời ngắn gọn, rõ ràng, dễ hiểu và có cấu trúc tốt
- Trích dẫn nguồn gốc thông tin khi có liên quan""",
        "user": """Câu hỏi: {query}

Bối cảnh:
{context}

HƯỚNG DẪN:
- Hãy cung cấp một câu trả lời toàn diện dựa trên bối cảnh ở trên
- **BẮTBUỘC: Trả lời bằng TIẾNG VIỆT, KHÔNG ĐƯỢC sử dụng ngôn ngữ khác**
- Nếu cần liệt kê, hãy sử dụng định dạng Việt (1., 2., 3., v.v.)"""
    },
    "english": {
        "system": """You are a helpful and knowledgeable AI assistant.

IMPORTANT INSTRUCTIONS:
- Provide responses in English
- Use the provided context to answer questions
- If context lacks information, state it clearly
- Keep answers concise, clear, and well-structured
- Cite sources when relevant""",
        "user": """Question: {query}

Context:
{context}

Please provide a comprehensive answer based on the context above. Answer in English."""
    }
}


def get_language_prompt(language: str, prompt_type: str, **kwargs) -> str:
    """
    Get language-specific prompt
    
    Args:
        language: Language code (vietnamese, english, etc.)
        prompt_type: "system" or "user"
        **kwargs: Variables to format in the prompt (query, context)
        
    Returns:
        Formatted prompt string
    """
    if language not in LANGUAGE_PROMPTS:
        logger.warning(f"Language '{language}' not supported, falling back to Vietnamese")
        language = "vietnamese"
    
    prompt = LANGUAGE_PROMPTS[language].get(prompt_type, "")
    
    try:
        return prompt.format(**kwargs)
    except KeyError as e:
        logger.error(f"Missing key in prompt format: {e}")
        return prompt


@dataclass
class GenerationResult:
    """Generated answer result"""
    answer: str
    sources: List[str]
    confidence: float
    model: str


class BaseLLMGenerator(ABC):
    """Base class for LLM generators"""
    
    @abstractmethod
    async def generate(self, query: str, context: str) -> GenerationResult:
        """Generate answer from query and context"""
        pass


class MockLLMGenerator(BaseLLMGenerator):
    """
    Mock LLM Generator - Returns realistic responses without API
    Perfect for testing and development
    Supports Vietnamese (Tiếng Việt) output
    """
    
    # Pre-defined responses for common queries - Vietnamese Language Support
    MOCK_RESPONSES = {
        "rag": {
            "answer": """RAG (Retrieval-Augmented Generation - Sinh thành Cải thiện Truy xuất) là một trong những bước tiến kiến trúc mang tính cách mạng nhất trong kỷ nguyên AI tạo sinh (Generative AI). Mô hình này giải quyết triệt để hai hạn chế cốt lõi của các Mô hình Ngôn ngữ Lớn (LLMs) truyền thống: **sự ảo tưởng thông tin (Hallucination)** và **sự thiếu hụt kiến thức thời gian thực (Knowledge Cut-off)**.

#### 🌀 1. Kiến trúc Hoạt động Chi tiết và Quy trình Xử lý (End-to-End Workflow)
Hệ thống RAG nâng cao hoạt động theo quy trình 4 giai đoạn khép kín cực kỳ chặt chẽ:
1. **Phân tích & Tối ưu hóa Truy vấn (Query Processing):** Nhận câu hỏi thô từ người dùng, đưa qua bộ Query Optimizer để sửa lỗi chính tả, chuẩn hóa cú pháp, và mở rộng từ khóa ngữ nghĩa bằng Tiếng Việt. Điều này giúp tối đa hóa khả năng khớp dữ liệu ở các bước sau.
2. **Truy xuất Thông tin Đa nguồn (Multi-Source Retrieval):** Hệ thống kích hoạt đồng thời hai kênh truy xuất độc lập:
   - **Vector Database (FAISS):** Quét qua kho tri thức cục bộ bằng tìm kiếm ngữ nghĩa thông qua độ tương đồng Cosine của các Vectơ biểu diễn (Embeddings).
   - **Tavily Search API:** Tìm kiếm thông tin trực tiếp từ Internet thời gian thực. Tavily được thiết kế chuyên biệt cho AI Agent, tự động loại bỏ quảng cáo, mã HTML rác và chỉ giữ lại văn bản sạch có độ tin cậy cao.
3. **Lọc, Chấm điểm & Xếp hạng lại (Filtering & Reranking):** Sử dụng các thuật toán chấm điểm thông minh để đánh giá độ liên quan của các tài liệu thu được. Các tài liệu có độ tương đồng cao nhất được xếp lên trên, loại bỏ hoàn toàn các thông tin rác hoặc trùng lặp.
4. **Tạo câu trả lời (Generative Answer):** Hệ thống tổng hợp câu hỏi gốc và ngữ cảnh chất lượng cao thu được thành một cấu trúc prompt tối ưu gửi đến LLM. LLM sẽ đóng vai trò là bộ não biên dịch và trình bày câu trả lời hoàn toàn dựa trên bằng chứng ngữ cảnh cung cấp (Grounding), đảm bảo tính chính xác tuyệt đối.

#### ⚖️ 2. So sánh RAG Truyền thống và RAG Nâng cao (Advanced RAG kết hợp Tavily API)
* **RAG Truyền thống (Naive RAG):** Chỉ dựa vào cơ sở dữ liệu tĩnh có sẵn. Gặp khó khăn lớn khi người dùng hỏi các thông tin mới phát sinh hoặc kiến thức chuyên ngành sâu mà doanh nghiệp chưa kịp cập nhật. Dễ bị loãng ngữ cảnh do lấy quá nhiều tài liệu nhiễu.
* **TAV-RAG Nâng cao:** Kết hợp sức mạnh của Vector DB nội bộ và khả năng tìm kiếm internet thời gian thực của Tavily API. Tavily cung cấp bộ lọc thông minh giúp tăng 40% tốc độ trích xuất thông tin chất lượng cao, giảm thiểu 85% chi phí token của LLM, và đảm bảo câu trả lời luôn cập nhật đến phút hiện tại.

#### 🎯 3. Các Chỉ số Đánh giá Hiệu năng RAG Cốt lõi
* **Precision@K (Độ chính xác):** Tỷ lệ các tài liệu thực sự liên quan trong số tài liệu được truy xuất.
* **Recall@K (Độ thu hồi):** Tỷ lệ các tài liệu liên quan được tìm thấy so với toàn bộ tài liệu liên quan có trong hệ thống.
* **F1-Score:** Giá trị trung bình điều hòa giữa Precision và Recall, phản ánh toàn diện chất lượng truy xuất.
* **Latency (Độ trễ):** Thời gian phản hồi từ lúc gửi câu hỏi đến lúc nhận được câu trả lời hoàn chỉnh.""",
            "sources": ["Tavily Search", "Vector DB", "RAG Architecture Paper"]
        },
        "embedding": {
            "answer": """Embedding (Vectơ hóa văn bản) là nền tảng toán học tối quan trọng của toàn bộ các hệ thống Tìm kiếm Ngữ nghĩa (Semantic Search) và Xử lý Ngôn ngữ Tự nhiên (NLP) hiện đại.

#### 🌀 1. Bản chất Toán học và Cơ chế Hoạt động
Về mặt toán học, Embedding là quá trình ánh xạ một thực thể ngôn ngữ (từ, cụm từ, câu hoặc toàn bộ tài liệu) thành một vectơ số thực trong không gian đa chiều có kích thước cố định (thường dao động từ 384 đến 1536 chiều, tùy thuộc vào mô hình như `all-MiniLM-L6-v2` hay `OpenAI text-embedding-3-small`).
* **Không gian Ngữ nghĩa (Semantic Space):** Các vectơ này được phân bố sao cho các khái niệm có ý nghĩa tương đồng nhau trong thực tế sẽ nằm gần nhau trong không gian hình học.
* **Độ tương đồng Cosine (Cosine Similarity):** Đo lường góc giữa hai vectơ. Góc càng nhỏ (độ tương đồng tiến gần về 1.0), ý nghĩa ngữ nghĩa của hai văn bản càng giống nhau, bất kể chúng có sử dụng chung từ vựng nào hay không.

#### 🛠️ 2. Quy trình Xử lý Embeddings trong Hệ thống Vector DB
1. **Phân đoạn văn bản (Text Chunking):** Cắt nhỏ tài liệu ban đầu thành các phần nhỏ (Chunks) có độ dài tối ưu (thường là 512 tokens) để bảo toàn tính tập trung của ngữ nghĩa.
2. **Tính toán Vectơ:** Đưa các đoạn văn bản qua mô hình nhúng (như Sentence-Transformers) để tạo ra các vectơ số tương ứng.
3. **Lưu trữ & Lập chỉ mục (Indexing):** Lưu vectơ cùng văn bản gốc vào cơ sở dữ liệu để phục vụ việc tính toán khoảng cách siêu tốc khi có truy vấn tìm kiếm của người dùng.

#### ⚡ 3. Các thách thức và Phương pháp tối ưu hóa Embeddings
* **Lỗi mất ngữ cảnh:** Chunk quá nhỏ có thể làm mất các thông tin liên kết quan trọng ở đoạn trước và sau. Biện pháp khắc phục là sử dụng cơ chế cửa sổ trượt (Sliding Window) hoặc lưu trữ siêu dữ liệu (Metadata).
* **Nhiễu thuật ngữ chuyên ngành:** Các mô hình embedding thông thường có thể không hiểu sâu sắc các từ viết tắt chuyên ngành. Giải pháp là kết hợp thêm bộ tối ưu hóa truy vấn (Query Expansion) hoặc thực hiện tinh chỉnh (Fine-tuning) mô hình embedding trên tập dữ liệu riêng biệt.""",
            "sources": ["Sentence Transformers", "FAISS Vector DB", "Semantic Search Fundamentals"]
        },
        "vector": {
            "answer": """Cơ sở dữ liệu vectơ (Vector Database) là loại hình cơ sở dữ liệu chuyên biệt được thiết kế để lưu trữ, quản lý và truy vấn hiệu quả các dữ liệu đa chiều, cụ thể là các vectơ biểu diễn ngữ nghĩa (Embeddings).

#### 🌀 1. Nguyên lý Hoạt động và Sự khác biệt với CSDL Truyền thống
* **CSDL Quan hệ (SQL) & Tài liệu (NoSQL):** Tìm kiếm thông tin dựa trên các khớp từ khóa chính xác (Keyword Matching) hoặc các chỉ mục quan hệ tĩnh. Rất kém hiệu quả khi người dùng nhập câu hỏi có tính ẩn dụ hoặc đồng nghĩa nhưng khác từ khóa viết.
* **CSDL Vectơ (Vector DB):** Sử dụng các thuật toán tìm kiếm lân cận gần nhất (Approximate Nearest Neighbor - ANN) để tìm kiếm các vectơ có khoảng cách hình học ngắn nhất so với vectơ của câu hỏi truy vấn. Điều này cho phép hệ thống hiểu được ý đồ thực sự của người dùng để trả về kết quả chính xác về mặt ý nghĩa ngữ nghĩa.

#### 🛠️ 2. Các thành phần và Thuật toán Lập chỉ mục Cốt lõi
1. **Thuật toán HNSW (Hierarchical Navigable Small World):** Xây dựng cấu trúc đồ thị đa tầng giúp tìm kiếm các điểm dữ liệu gần nhất với độ phức tạp thời gian cực thấp (O(log N)), cực kỳ phù hợp cho các tập dữ liệu hàng triệu văn bản.
2. **Phân cụm IVF (Inverted File Index):** Chia không gian vectơ thành các phân vùng lớn để giới hạn phạm vi tìm kiếm, giúp tăng tốc độ phản hồi đáng kể.
3. **Lượng tử hóa vectơ (Scalar Quantization):** Nén kích thước của các vectơ số thực để tiết kiệm bộ nhớ RAM và tăng băng thông tính toán khoảng cách.

#### 🎯 3. Vai trò của Vector DB trong Hệ thống TAV-RAG
Trong hệ thống TAV-RAG của chúng tôi, Vector DB đóng vai trò làm **Kho tri thức dài hạn (Long-term Memory)**. Nó lưu giữ toàn bộ dữ liệu nội bộ đã được cấu trúc hóa, cho phép hệ thống truy xuất các dữ liệu lịch sử cực kỳ nhanh chóng (dưới 50ms) trước khi quyết định gọi thêm API tìm kiếm thời gian thực Tavily Search để bổ sung dữ liệu mới phát sinh.""",
            "sources": ["Vector Databases Deep Dive", "HNSW Algorithm Paper", "FAISS Library"]
        },
        "tavily": {
            "answer": """Tavily Search API là công cụ tìm kiếm internet thế hệ mới, được tối ưu hóa và thiết kế chuyên biệt cho các tác nhân trí tuệ nhân tạo (AI Agents) và các hệ thống RAG (Retrieval-Augmented Generation) tiên tiến.

#### 🌀 1. Điểm vượt trội của Tavily so với các Công cụ Tìm kiếm Truyền thống (Google, Bing Search)
* **Google/Bing Search:** Trả về kết quả là danh sách các trang web hướng tới người dùng đọc, chứa đầy quảng cáo, mã theo dõi, mã HTML phức tạp và các văn bản điều hướng dư thừa. Điều này gây lãng phí nghiêm trọng cửa sổ bối cảnh (Context Window) của LLM và làm tăng chi phí token lên gấp 5-10 lần.
* **Tavily Search API:** Tự động thu thập, trích xuất văn bản thô, làm sạch triệt để dữ liệu rác, và tái xếp hạng nội dung dựa trên độ liên quan trực tiếp với câu hỏi của AI. Dữ liệu trả về dưới dạng JSON cực kỳ gọn gàng, chứa chính xác các đoạn văn bản hữu ích cùng nguồn URL tham chiếu đáng tin cậy.

#### 🛠️ 2. Quy trình tích hợp Tavily trong Kiến trúc TAV-RAG
1. **Nhận diện ý định định nghĩa (Intent Detection):** Bộ lọc thông minh tự động phát hiện xem câu hỏi có cần thông tin cập nhật thời gian thực hoặc định nghĩa chuyên sâu hay không.
2. **Gửi truy vấn song song:** Gọi đồng thời API Tavily để lấy thông tin từ môi trường internet toàn cầu kết hợp quét Vector DB nội bộ.
3. **Tổng hợp dữ liệu thông minh:** Các kết quả tìm kiếm từ internet của Tavily được chấm điểm và lọc nhiễu chéo, đảm bảo loại bỏ các trang web spam hoặc tin giả (Fake News) trước khi nạp làm ngữ cảnh cho LLM sinh câu trả lời.

#### 🎯 3. Các tham số cấu hình tối ưu của Tavily Search
* **Search Depth (Độ sâu tìm kiếm):** Gồm hai chế độ `basic` (nhanh, tiết kiệm) và `advanced` (tìm kiếm sâu, phân tích nhiều nguồn chất lượng cao).
* **Include Raw Content:** Cho phép lấy toàn bộ nội dung văn bản thô của trang web để hệ thống tự phân tích chuyên sâu.
* **Max Results:** Khống chế số lượng trang web trả về (thường tối ưu ở mức 3 đến 5 nguồn chất lượng nhất) để giữ độ trễ hệ thống ở mức dưới 500ms.""",
            "sources": ["Tavily API Documentation", "AI Search Optimization", "TAV-RAG Integration"]
        },
        "performance": {
            "answer": """Tối ưu hóa hiệu năng (Performance Tuning) là khâu then chốt để chuyển đổi một hệ thống RAG từ dạng thử nghiệm (Prototype) sang sản phẩm thương mại hoạt động ổn định trong môi trường thực tế với hàng triệu người dùng.

#### 🌀 1. Các Chiến lược Tối ưu hóa Cấp độ Dữ liệu và Truy xuất
1. **Thiết lập Kích thước Chunk Động (Dynamic Chunking):** Thay vì cắt văn bản cứng nhắc ở một số lượng ký tự cố định, hệ thống RAG nâng cao sử dụng kích thước chunk động dựa trên cấu trúc ngữ pháp (dấu câu, phân đoạn) để bảo toàn ngữ nghĩa của từng luận điểm.
2. **Xếp hạng lại đa giai đoạn (Multi-stage Reranking):** 
   - *Giai đoạn 1:* Sử dụng thuật toán tìm kiếm vector nhanh để lấy ra top 50 tài liệu ứng viên.
   - *Giai đoạn 2:* Dùng mô hình Cross-Encoder chuyên biệt để chấm điểm chi tiết và chọn lọc ra top 3 đến top 5 tài liệu liên quan nhất gửi cho LLM. Điều này giảm 60% kích thước context nạp vào LLM mà không làm giảm độ chính xác.

#### 🛠️ 2. Chiến lược Tối ưu hóa Bộ nhớ đệm (Caching Strategy)
* **Hybrid Caching:** Kết hợp bộ nhớ đệm nhanh RAM (Memory Cache) cho các truy vấn lặp lại liên tục trong thời gian ngắn và bộ nhớ đệm tệp tin (File Cache) bền vững để lưu trữ lâu dài.
* **Tác động thực tế:** Giảm thiểu tới 50% chi phí gọi API tìm kiếm và API LLM bên ngoài, đồng thời giảm độ trễ phản hồi cho các câu hỏi trùng lặp từ mức ~1.5 giây xuống mức chỉ từ 2-5ms (tăng tốc độ lên tới 300 lần!).

#### 🎯 3. Tối ưu hóa Quy trình Xử lý Hàng loạt (Batch Processing)
* **Xử lý bất đồng bộ (Asyncio Concurrent):** Cho phép hệ thống xử lý song song hàng trăm yêu cầu của người dùng cùng lúc mà không bị nghẽn luồng (Blocking).
* **Nhóm câu hỏi trùng lặp (Query Grouping):** Tự động phát hiện và gộp các câu hỏi có cùng ý nghĩa ngữ nghĩa để xử lý một lần duy nhất, chia sẻ kết quả cho nhiều người dùng đồng thời, giúp tiết kiệm tối đa tài nguyên tính toán.""",
            "sources": ["RAG Production Best Practices", "Latency Optimization Guide", "TAV-RAG Benchmarks"]
        },
        "cache": {
            "answer": """Bộ nhớ đệm (Caching) là giải pháp tối ưu hóa hiệu năng cực kỳ quan trọng, giúp các hệ thống RAG tiết kiệm chi phí vận hành, giảm tải cho API bên ngoài, và cải thiện trải nghiệm người dùng bằng cách giảm độ trễ phản hồi xuống mức gần như bằng 0.

#### 🌀 1. Phân loại Bộ nhớ đệm trong Hệ thống TAV-RAG
Hệ thống của chúng tôi áp dụng cơ chế **Lưu trữ đệm Lai (Hybrid Caching)** kết hợp ưu điểm của hai tầng lưu trữ:
1. **Bộ nhớ đệm RAM (In-Memory Cache):** Lưu trữ dữ liệu trực tiếp trên bộ nhớ trong của máy chủ.
   - *Đặc điểm:* Tốc độ đọc/ghi siêu tốc (dưới 1ms).
   - *Hạn chế:* Dữ liệu sẽ biến mất khi máy chủ khởi động lại hoặc gặp sự cố nguồn điện.
2. **Bộ nhớ đệm Tệp tin (Persistent File Cache):** Lưu trữ dữ liệu đã được mã hóa dưới dạng tệp tin cục bộ trong thư mục `cache/`.
   - *Đặc điểm:* Bền vững, dữ liệu được bảo toàn vĩnh viễn qua các lần chạy hoặc khởi động lại hệ thống.
   - *Tốc độ:* Nhanh vượt trội so với việc thực hiện lại toàn bộ quy trình RAG (quét vector, gọi API tìm kiếm, gọi LLM).

#### 🛠️ 2. Nguyên lý Hoạt động của Cơ chế So khớp Cache Ngữ nghĩa
Thay vì chỉ so khớp khớp chính xác từng ký tự câu hỏi (dễ bị bỏ sót khi câu hỏi chỉ khác một dấu cách hoặc viết hoa):
* **Tính toán Hash MD5:** Hệ thống chuẩn hóa câu hỏi (loại bỏ khoảng trắng thừa, chuyển về chữ thường) và tính toán mã băm MD5 duy nhất làm khóa Cache (Cache Key).
* **Độ tương đồng ngữ nghĩa (Semantic Caching):** Trong các phiên bản mở rộng, hệ thống sử dụng embedding của câu hỏi để so sánh độ tương đồng với các khóa đã lưu trong cache. Nếu câu hỏi mới có độ tương đồng ngữ nghĩa trên 95% so với một câu hỏi đã xử lý trước đó, hệ thống sẽ trả về ngay kết quả đã lưu.

#### 🎯 3. Tác động của Caching đến Chi phí và Hiệu suất
* **Tiết kiệm tài nguyên:** Mỗi lượt trúng bộ nhớ đệm (Cache Hit) giúp tiết kiệm 100% chi phí gọi API Tavily Search và API LLM bên ngoài.
* **Thời gian phản hồi siêu tốc:** Độ trễ giảm từ ~1000ms xuống dưới 5ms, mang lại trải nghiệm tương tác mượt mà và chuyên nghiệp nhất cho người sử dụng.""",
            "sources": ["Caching Patterns in Distributed Systems", "TAV-RAG Cache Architecture"]
        }
    }
    
    def __init__(self, model_name: str = "gpt-4-mock", language: str = "vietnamese"):
        """
        Initialize mock LLM generator
        
        Args:
            model_name: Model name (for logging/tracking)
            language: Output language (vietnamese, english, etc.)
        """
        self.model_name = model_name
        self.language = language
        logger.info(f"Mock LLM Generator initialized: {model_name}, Language: {language.upper()}")
    
    async def generate(self, query: str, context: str) -> GenerationResult:
        """
        Generate context-aware mock answer with enhanced relevance
        
        Args:
            query: User query
            context: Retrieved context (formatted with sources)
            
        Returns:
            GenerationResult with answer in Vietnamese
        """
        import asyncio
        import re
        
        # Simulate API latency
        await asyncio.sleep(0.2)
        
        # 1. Tìm câu trả lời mẫu cơ bản
        predefined_answer = None
        sources = ["Mock Generation"]
        
        for keyword, response in self.MOCK_RESPONSES.items():
            if keyword.lower() in query.lower():
                predefined_answer = response["answer"]
                sources = response["sources"].copy()
                break
        
        # 2. Phân tích bối cảnh (context) được truy xuất dưới dạng cấu trúc [Tài liệu X]
        doc_blocks = re.split(r'\[Tài liệu \d+\]', context)
        doc_blocks = [db.strip() for db in doc_blocks if db.strip()]
        
        # Trích xuất nguồn thực tế từ các tài liệu tìm được
        context_sources = []
        parsed_docs = []
        
        for db in doc_blocks:
            lines = db.split('\n')
            title = ""
            url = ""
            source_info = ""
            content_lines = []
            
            for line in lines:
                if line.startswith("Tiêu đề:"):
                    title = line.replace("Tiêu đề:", "").strip()
                elif line.startswith("Nguồn URL:"):
                    url = line.replace("Nguồn URL:", "").strip()
                elif line.startswith("Nguồn trích xuất:"):
                    source_info = line.replace("Nguồn trích xuất:", "").strip()
                elif line.startswith("Nội dung:"):
                    content_lines.append(line.replace("Nội dung:", "").strip())
                else:
                    if content_lines:
                        content_lines.append(line.strip())
            
            doc_content = " ".join(content_lines).strip()
            if not doc_content:
                doc_content = db
            
            parsed_docs.append({
                "title": title,
                "url": url,
                "source": source_info or "Hệ thống tri thức",
                "content": doc_content
            })
            
            if source_info and source_info not in context_sources:
                context_sources.append(source_info)
                
        if context_sources:
            sources = list(set(sources + context_sources))[:4]
            
        # 3. Tạo câu trả lời đầy đủ, chi tiết và có cấu trúc cao bằng Tiếng Việt
        answer_parts = []
        
        # Tiêu đề chuyên nghiệp
        answer_parts.append(f"### 🎯 BÁO CÁO PHÂN TÍCH VÀ CÂU TRẢ LỜI CHI TIẾT\n**Truy vấn:** *\"{query}\"*")
        
        # Định nghĩa & Giới thiệu
        answer_parts.append("#### 1. Định nghĩa và Khái niệm Cốt lõi")
        if predefined_answer:
            answer_parts.append(predefined_answer)
        else:
            # Tìm đoạn văn giới thiệu tốt nhất từ context
            intro_candidate = ""
            if parsed_docs:
                for doc in parsed_docs:
                    c = doc["content"]
                    if any(kw in c.lower() for kw in ["là gì", "định nghĩa", "khái niệm", "is a", "combines"]):
                        intro_candidate = c
                        break
                if not intro_candidate:
                    intro_candidate = parsed_docs[0]["content"]
                
            if intro_candidate:
                answer_parts.append(intro_candidate)
            else:
                answer_parts.append(f"Dựa trên dữ liệu truy xuất hệ thống, **{query}** được hiểu là một chủ thể/khái niệm quan trọng trong lĩnh vực công nghệ thông tin và kiến trúc RAG, hỗ trợ đắc lực cho các mô hình ngôn ngữ lớn (LLM) trong việc xử lý ngữ nghĩa và nâng cao độ chính xác của thông tin đầu ra.")
        
        # Chi tiết hệ thống & Phân tích chuyên sâu
        answer_parts.append("#### 2. Chi tiết Kỹ thuật và Bằng chứng Ngữ cảnh")
        
        has_details = False
        if parsed_docs:
            for i, doc in enumerate(parsed_docs[:5], 1):
                # Loại bỏ đoạn văn trùng lặp với định nghĩa hoặc quá ngắn
                c = doc["content"]
                if predefined_answer and c in predefined_answer:
                    continue
                if len(c) < 30:
                    continue
                
                title_part = f" - *\"{doc['title']}\"*" if doc['title'] else ""
                url_part = f" (Xem thêm: {doc['url']})" if doc['url'] else ""
                
                answer_parts.append(
                    f"##### 📄 Tài liệu tham chiếu {i} [{doc['source']}]{title_part}\n"
                    f"* **Nội dung:** {c}\n"
                    f"* **Đánh giá hệ thống:** Tài liệu có độ tương hợp cao, cung cấp b bằng chứng xác đáng cho câu hỏi của người dùng.{url_part}"
                )
                has_details = True
                
        if not has_details:
            answer_parts.append("- **Đặc điểm chính:** Tối ưu hóa hiệu năng bằng cách kết hợp truy xuất ngữ nghĩa động với nguồn dữ liệu thời gian thực có độ chính xác cao.\n- **Kiến trúc:** Tích hợp bộ lọc thông minh giúp loại bỏ nhiễu và chỉ giữ lại bối cảnh có điểm liên quan lớn nhất.\n- **Khả năng mở rộng:** Dễ dàng tương thích với các cơ sở dữ liệu vector tiên tiến và các API tìm kiếm hàng đầu như Tavily Search.")

        # Tác động & Ứng dụng thực tế
        answer_parts.append("#### 3. Tác động và Ứng dụng trong Mô hình TAV-RAG")
        answer_parts.append(
            f"Trong hệ thống **TAV-RAG (Tavily Enhanced Retrieval-Augmented Generation)**, việc tối ưu hóa và trả lời toàn diện cho truy vấn này đóng vai trò quyết định giúp:\n"
            f"1. **Tăng độ tin cậy:** Tránh hoàn toàn lỗi ảo tưởng (hallucination) thường gặp ở các LLM thông thường nhờ cơ chế kiểm chứng chéo thông tin.\n"
            f"2. **Tối ưu hóa tài nguyên:** Giảm thiểu độ trễ xử lý (Latency) nhờ hệ thống lưu trữ đệm (Cache Manager) và cấu trúc phân tích câu lệnh thông minh.\n"
            f"3. **Tương thích đa ngôn ngữ:** Tự động phát hiện và tối ưu hóa câu trả lời sang tiếng Việt chuẩn, đáp ứng chính xác nhu cầu tra cứu học thuật lẫn ứng dụng thực tiễn."
        )

        answer = "\n\n".join(answer_parts)
        
        result = GenerationResult(
            answer=answer,
            sources=sources,
            confidence=0.88,
            model=self.model_name
        )
        
        logger.info(f"Generated comprehensive answer for query: {query[:50]}... (sources: {len(sources)})")
        return result
    
    def generate_sync(self, query: str, context: str) -> GenerationResult:
        """Synchronous generation wrapper"""
        import asyncio
        return asyncio.run(self.generate(query, context))


class OpenAILLMGenerator(BaseLLMGenerator):
    """
    Real OpenAI LLM Generator
    Requires OPENAI_API_KEY environment variable
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-4-turbo", language: str = "vietnamese"):
        """
        Initialize OpenAI generator
        
        Args:
            api_key: OpenAI API key
            model: Model name (gpt-4, gpt-3.5-turbo, etc)
            language: Output language (vietnamese, english, etc.)
        """
        if not api_key:
            raise ValueError("OpenAI API key required")
        
        try:
            import openai
            openai.api_key = api_key
            self.client = openai.OpenAI(api_key=api_key)
        except ImportError:
            raise ImportError("openai package required. Install: pip install openai")
        
        self.model = model
        self.api_key = api_key
        self.language = language
        logger.info(f"OpenAI LLM Generator initialized: {model}, Language: {language.upper()}")
    
    async def generate(self, query: str, context: str) -> GenerationResult:
        """
        Generate answer using OpenAI API
        
        Args:
            query: User query
            context: Retrieved context
            
        Returns:
            GenerationResult with answer
        """
        import asyncio
        
        # Run in thread pool since openai is sync
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, 
            self._generate_sync,
            query,
            context
        )
        
        return result
    
    def _generate_sync(self, query: str, context: str) -> GenerationResult:
        """Synchronous generation (wrapped for async execution)"""
        
        # Get language-specific prompts
        system_prompt = get_language_prompt(self.language, "system")
        user_message = get_language_prompt(self.language, "user", query=query, context=context)
        
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message}
                ],
                temperature=0.7,
                max_tokens=500
            )
            
            answer = response.choices[0].message.content
            
            result = GenerationResult(
                answer=answer,
                sources=["OpenAI", "Retrieved Context"],
                confidence=0.9,
                model=self.model
            )
            
            logger.info(f"Generated answer using OpenAI ({self.model}) in {self.language.upper()}")
            return result
            
        except Exception as e:
            logger.error(f"OpenAI generation error: {e}")
            raise
    
    def generate_sync(self, query: str, context: str) -> GenerationResult:
        """Synchronous generation wrapper"""
        return self._generate_sync(query, context)


def create_llm_generator(use_mock: bool = True, api_key: str = None, model: str = "gpt-4-turbo", language: str = "vietnamese"):
    """
    Factory function to create LLM generator
    
    Args:
        use_mock: Use mock generator (True) or real OpenAI (False)
        api_key: OpenAI API key (required if use_mock=False)
        model: Model name
        language: Output language (vietnamese, english, etc.)
        
    Returns:
        LLM generator instance
    """
    if use_mock:
        logger.info(f"Using Mock LLM Generator with language: {language.upper()}")
        return MockLLMGenerator(model, language=language)
    else:
        logger.info(f"Using OpenAI LLM Generator with language: {language.upper()}")
        return OpenAILLMGenerator(api_key=api_key, model=model, language=language)

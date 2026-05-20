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
            "answer": "RAG (Sinh thành Cải thiện Truy xuất) kết hợp hai kỹ thuật: trước tiên, nó truy xuất các tài liệu có liên quan từ cơ sở kiến thức bằng tìm kiếm ngữ nghĩa, sau đó sử dụng các tài liệu này làm bối cảnh để tạo ra câu trả lời chính xác hơn. Phương pháp này giảm đáng kể việc tạo ra thông tin sai lệch và đảm bảo các câu trả lời được dựa trên thông tin có thật. RAG đặc biệt hữu ích khi xử lý kiến thức chuyên ngành hoặc thông tin gần đây có thể không nằm trong dữ liệu huấn luyện của mô hình ngôn ngữ.",
            "sources": ["Tavily Search", "Vector DB"]
        },
        "embedding": {
            "answer": "Các embedding văn bản là những biểu diễn số của văn bản nắm bắt ý nghĩa ngữ nghĩa. Chúng chuyển đổi các từ hoặc câu thành các vectơ (mảng số) trong không gian đa chiều, nơi các văn bản tương tự gần nhau hơn. Các mô hình embedding phổ biến bao gồm BERT, GPT-2 embeddings và sentence-transformers. Những embeddings này rất cần thiết cho: tìm kiếm tương tự, phân cụm, phân loại và các ứng dụng tìm kiếm ngữ nghĩa.",
            "sources": ["Tavily Search", "Vector DB"]
        },
        "machine learning": {
            "answer": "Học máy là một tập hợp con của trí tuệ nhân tạo cho phép các hệ thống học hỏi và cải thiện từ kinh nghiệm mà không cần lập trình rõ ràng. Nó liên quan đến việc huấn luyện các thuật toán trên dữ liệu để nhận biết các mẫu và đưa ra dự đoán. Các loại chính bao gồm: học có giám sát (với dữ liệu có nhãn), học không giám sát (tìm kiếm mẫu) và học tăng cường (học từ phần thưởng). Các ứng dụng trải rộng từ các hệ thống khuyến nghị đến thị giác máy tính.",
            "sources": ["Tavily Search"]
        },
        "performance": {
            "answer": "Tối ưu hóa hiệu suất trong hệ thống AI liên quan đến nhiều chiến lược: 1) Giảm kích thước mô hình thông qua lượng tử hóa hoặc tinh lọc, 2) Triển khai bộ nhớ đệm cho các truy vấn thường xuyên, 3) Sử dụng xử lý hàng loạt cho nhiều yêu cầu, 4) Tối ưu hóa các truy vấn cơ sở dữ liệu và lập chỉ mục, 5) Triển khai các hoạt động không đồng bộ cho các nhiệm vụ bị ràng buộc I/O. Đối với các hệ thống RAG cụ thể, hãy tối ưu hóa bằng cách cải thiện chất lượng truy xuất, giảm kích thước cửa sổ bối cảnh và triển khai xếp hạng hiệu quả.",
            "sources": ["Vector DB", "System Design"]
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
        
        # Simulate API latency
        await asyncio.sleep(0.2)
        
        # Find matching response from pre-defined responses
        answer = None
        sources = ["Mock Generation"]
        
        for keyword, response in self.MOCK_RESPONSES.items():
            if keyword.lower() in query.lower():
                answer = response["answer"]
                sources = response["sources"]
                break
        
        # Enhanced context-aware response if no pre-defined match
        if not answer:
            # Extract sources from context
            context_sources = []
            source_patterns = [r'\[.*?\]', r'\(.*?\)']
            for pattern in source_patterns:
                matches = re.findall(pattern, context)
                context_sources.extend(matches)
            
            # Build context-aware answer
            context_preview = context.split('\n')[0][:150] if context else ""
            
            answer = (
                f"Dựa trên thông tin được truy xuất, tôi có thể cung cấp nhận xét sau về '{query}':\n\n"
                f"Thông tin chính: {context_preview}\n\n"
                f"Theo bối cảnh được cung cấp, điều này tiếp tục với sâu hơn là: "
                f"Hệ thống RAG của chúng tôi lấy được tài liệu liên quan nhất và sử dụng chúng để tạo ra "
                f"một câu trả lời chính xác dựa trên kiến thức hiện tại. Điều này đảm bảo rằng "
                f"các câu trả lời được dựa trên thông tin có thật thay vì học phỏng đoán."
            )
            
            if context_sources:
                sources = context_sources[:3]  # Take first 3 sources
            else:
                sources = ["Tạo tác dựa trên bối cảnh", "Truy xuất đa giai đoạn"]
        
        result = GenerationResult(
            answer=answer,
            sources=sources,
            confidence=0.82 if not answer else 0.85,  # Slightly lower confidence for generated answers
            model=self.model_name
        )
        
        logger.info(f"Generated answer for query: {query[:50]}... (sources: {len(sources)})")
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

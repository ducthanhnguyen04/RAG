"""
Query Optimizer - Improve query quality before retrieval
Includes: query rewriting, keyword extraction, intent detection
Enhanced with: Vietnamese to English translation, context enhancement
"""
from typing import List, Dict, Tuple
from enum import Enum
import re
from utils.logger import setup_logger
from utils.text_processor import extract_keywords, clean_text, normalize_vietnamese_text

logger = setup_logger(__name__)

# Vietnamese to English translation map for search optimization
VIETNAMESE_TO_ENGLISH_PHRASES = {
    # AI/ML terms
    'trí tuệ nhân tạo': 'artificial intelligence',
    'máy học': 'machine learning',
    'mạng nơ-ron': 'neural network',
    'xử lý ngôn ngữ tự nhiên': 'natural language processing',
    'học sâu': 'deep learning',
    'tạo sinh': 'generative',
    'sinh tạo': 'generative',
    
    # RAG specific
    'retrieval augmented generation': 'retrieval augmented generation',
    'tăng cường truy xuất': 'retrieval augmented generation',
    'rag': 'retrieval augmented generation',
    'tavily': 'tavily search engine',
    
    # Common question patterns
    'là gì': 'definition',
    'định nghĩa': 'definition',
    'khái niệm': 'concept',
    'cách': 'how to',
    'hướng dẫn': 'guide tutorial',
    'bước': 'steps',
    'tại sao': 'why',
    'lý do': 'reason',
    'so sánh': 'comparison',
    'khác biệt': 'difference',
    'giống nhau': 'similarity',
    'mới nhất': 'latest recent',
    'tin tức': 'news',
    'cập nhật': 'update',
    
    # Technical terms
    'mô hình': 'model',
    'dữ liệu': 'data',
    'phân tích': 'analysis',
    'tìm kiếm': 'search',
    'thuật toán': 'algorithm',
    'lập trình': 'programming',
    'phần mềm': 'software',
    'ứng dụng': 'application',
    'hệ thống': 'system',
    'cơ sở dữ liệu': 'database',
}


class QueryIntent(Enum):
    """Types of query intents"""
    WHAT = "definition"        # What is X?
    HOW = "how_to"            # How to do X?
    WHY = "reason"            # Why is X?
    COMPARE = "comparison"    # Compare X vs Y
    RECENT = "recent_info"    # Latest about X
    GENERAL = "general"       # General query


class QueryOptimizer:
    """
    Optimize queries for better retrieval
    Techniques:
    - Intent detection
    - Keyword extraction
    - Query rewriting
    - Normalization
    """
    
    def __init__(self):
        """Initialize query optimizer"""
        self.logger = logger
    
    def detect_intent(self, query: str) -> QueryIntent:
        """
        Detect query intent
        
        Args:
            query: Input query
            
        Returns:
            QueryIntent enum
        """
        query_lower = query.lower().strip()
        
        # Check for specific patterns
        if query_lower.startswith(('what is', 'what\'s', 'what are')):
            return QueryIntent.WHAT
        elif query_lower.startswith(('how to', 'how can', 'how do')):
            return QueryIntent.HOW
        elif query_lower.startswith(('why', 'why is', 'why do', 'why does')):
            return QueryIntent.WHY
        elif 'vs' in query_lower or 'versus' in query_lower or 'compare' in query_lower:
            return QueryIntent.COMPARE
        elif any(word in query_lower for word in ['latest', 'recent', 'new', '2024', '2025', '2026']):
            return QueryIntent.RECENT
        else:
            return QueryIntent.GENERAL
    
    def normalize_query(self, query: str) -> str:
        """
        Normalize query text
        
        Args:
            query: Input query
            
        Returns:
            Normalized query
        """
        # Remove extra whitespace
        query = re.sub(r'\s+', ' ', query).strip()
        
        # Remove common question marks at end
        query = query.rstrip('?!.,')
        
        # Convert to lowercase for processing
        # But preserve for actual search
        return query
    
    def _translate_to_english(self, query: str) -> str:
        """
        Translate Vietnamese phrases to English for better search
        Giúp tìm kiếm tiếng Anh được cải thiện
        
        Args:
            query: Input query (possibly Vietnamese)
            
        Returns:
            Query with Vietnamese phrases translated to English
        """
        result = query
        
        # Translate phrases (longer phrases first for better matching)
        for vietnamese, english in sorted(VIETNAMESE_TO_ENGLISH_PHRASES.items(), 
                                        key=lambda x: len(x[0]), reverse=True):
            # Use word boundaries to avoid partial replacements
            pattern = r'\b' + re.escape(vietnamese) + r'\b'
            result = re.sub(pattern, english, result, flags=re.IGNORECASE)
        
        return result
    
    def extract_entities(self, query: str) -> List[str]:
        """
        Extract main entities from query
        
        Args:
            query: Input query
            
        Returns:
            List of entities
        """
        # Remove common words
        stop_words = {
            'what', 'how', 'why', 'where', 'when', 'which', 'who',
            'is', 'are', 'am', 'be', 'been', 'being',
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
            'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through',
            'vs', 'versus', 'compare', 'versus'
        }
        
        words = [w for w in query.lower().split() 
                if w not in stop_words and len(w) > 2]
        
        return words
    
    def generate_variations(self, query: str) -> List[str]:
        """
        Generate query variations for better coverage
        Enhanced with more intelligent variations for semantic search
        
        Args:
            query: Original query
            
        Returns:
            List of query variations optimized for retrieval
        """
        variations = [query]
        
        intent = self.detect_intent(query)
        entities = self.extract_entities(query)
        keywords = extract_keywords(query, num_keywords=5)
        
        # Generate variations based on intent (Vietnamese-optimized)
        if intent == QueryIntent.WHAT:
            # "What is X" → enhanced variations
            entity_str = ' '.join(entities)
            main_keyword = entities[0] if entities else ""
            
            variations.extend([
                f"{entity_str} là gì",  # Vietnamese variant
                f"khái niệm {entity_str}",
                f"định nghĩa {entity_str}",
                f"{entity_str} definition",
                f"explain {entity_str}",
                f"understanding {entity_str}",
                entity_str,
                main_keyword if main_keyword else entity_str,
            ])
        
        elif intent == QueryIntent.HOW:
            # "How to X" → enhanced variations
            entity_str = ' '.join(entities)
            variations.extend([
                f"cách {entity_str}",  # Vietnamese variant
                f"hướng dẫn {entity_str}",
                f"bước thực hiện {entity_str}",
                f"{entity_str} tutorial",
                f"{entity_str} guide",
                f"{entity_str} how-to",
                f"steps for {entity_str}",
                f"implement {entity_str}",
                entity_str,
            ])
        
        elif intent == QueryIntent.COMPARE:
            # "X vs Y" → enhanced comparison variations
            if len(entities) >= 2:
                entity1, entity2 = entities[0], entities[1]
                variations.extend([
                    f"{entity1} so với {entity2}",  # Vietnamese variant
                    f"so sánh {entity1} và {entity2}",
                    f"comparison of {entity1} vs {entity2}",
                    f"{entity1} versus {entity2}",
                    f"difference between {entity1} and {entity2}",
                    f"similarities and differences {entity1} {entity2}",
                ])
        
        elif intent == QueryIntent.WHY:
            # "Why X" → enhanced variations
            entity_str = ' '.join(entities)
            variations.extend([
                f"tại sao {entity_str}",  # Vietnamese variant
                f"lý do {entity_str}",
                f"nguyên nhân {entity_str}",
                f"why is {entity_str} important",
                f"importance of {entity_str}",
                f"benefits of {entity_str}",
            ])
        
        elif intent == QueryIntent.RECENT:
            # "Recent X" → time-aware variations
            entity_str = ' '.join(entities)
            variations.extend([
                f"{entity_str} 2025 2026",  # Recent years
                f"recent {entity_str}",
                f"latest {entity_str}",
                f"tin tức {entity_str}",  # Vietnamese variant
                f"cập nhật {entity_str}",
                f"mới nhất về {entity_str}",
            ])
        
        # Add semantic expansion with keywords
        if keywords:
            # Create multi-keyword queries for better semantic matching
            main_keywords = keywords[:2]
            if main_keywords:
                variations.append(' '.join(main_keywords))
            if len(keywords) >= 3:
                variations.append(' '.join(keywords[:3]))
        
        # Add contextual variations
        if len(entities) > 0:
            main_entity = entities[0]
            variations.extend([
                f"information about {main_entity}",
                f"facts about {main_entity}",
                f"overview {main_entity}",
                main_entity,
            ])
        
        # Remove duplicates while preserving order
        seen = set()
        unique_variations = []
        for v in variations:
            v_norm = v.lower().strip()
            if v_norm not in seen and v_norm:  # Skip empty strings
                seen.add(v_norm)
                unique_variations.append(v)
        
        self.logger.info(f"Generated {len(unique_variations)} optimized query variations")
        return unique_variations
    
    def optimize_query(self, query: str) -> Dict:
        """
        Full query optimization with Vietnamese-to-English translation
        - Dịch tiếng Việt sang tiếng Anh để tìm kiếm tốt hơn
        - Chuẩn hóa tiếng Việt nâng cao
        - Mở rộng từ khóa thông minh
        - Sinh nhiều biến thể truy vấn tối ưu
        - Phát hiện chủ đề và bối cảnh
        
        Args:
            query: Input query
            
        Returns:
            Dictionary with comprehensive optimization results
        """
        # Step 1: Dịch tiếng Việt sang tiếng Anh ưu tiên
        translated_query = self._translate_to_english(query)
        
        # Step 2: Chuẩn hóa tiếng Việt ưu tiên
        try:
            query_norm = normalize_vietnamese_text(translated_query)
        except Exception:
            query_norm = translated_query
        normalized = self.normalize_query(query_norm)
        
        # Step 3: Phát hiện ý định
        intent = self.detect_intent(normalized)
        
        # Step 4: Trích xuất các thực thể chính
        entities = self.extract_entities(normalized)
        
        # Step 5: Trích xuất từ khóa (tăng số lượng)
        keywords = extract_keywords(normalized, num_keywords=10)
        
        # Step 6: Sinh các biến thể truy vấn tối ưu
        variations = self.generate_variations(normalized)
        
        # Step 7: Thêm so sánh từ đồng nghĩa
        expanded_query = normalized
        for syn_short, syn_long in QueryRewriter.SYNONYMS.items():
            pattern = r'\b' + re.escape(syn_short) + r'\b'
            if re.search(pattern, expanded_query.lower()):
                expanded_query = normalized.replace(syn_short, syn_long)
                break
        
        if expanded_query != normalized:
            variations.append(expanded_query)
        
        # Step 8: Phân tích độ dài và độ phức tạp truy vấn
        query_length = len(normalized.split())
        is_complex = query_length > 5 or intent in [QueryIntent.COMPARE, QueryIntent.WHY]
        
        # Step 9: Loại bỏ trùng lặp và sắp xếp theo mức độ quan trọng
        seen = set()
        unique_variations = []
        for v in variations:
            v_norm = v.lower().strip()
            if v_norm not in seen and v_norm:
                seen.add(v_norm)
                unique_variations.append(v)
        
        # Sắp xếp: truy vấn gốc trước, sau đó là biến thể chuẩn hóa
        if unique_variations and unique_variations[0].lower() != normalized.lower():
            unique_variations.insert(0, normalized)
        
        # Step 10: Xác định mô tả tách biệt cho ngôn ngữ
        intent_descriptions = {
            "definition": "Định nghĩa và giải thích",
            "how_to": "Hướng dẫn và bước thực hiện",
            "reason": "Lý do và tầm quan trọng",
            "comparison": "So sánh và khác biệt",
            "recent_info": "Thông tin gần đây",
            "general": "Thông tin chung"
        }
        
        result = {
            "original": query.strip(),
            "translated": translated_query,  # Add translated version
            "normalized": normalized,
            "intent": intent.value,
            "intent_description": intent_descriptions.get(intent.value, "Không xác định"),
            "entities": entities,
            "keywords": keywords,
            "query_complexity": "complex" if is_complex else "simple",
            "query_length": query_length,
            "variations": unique_variations,
            "main_query": unique_variations[0] if unique_variations else normalized,
            "semantic_expansion": expanded_query,
            "confidence": 0.95 if len(entities) > 0 else 0.7  # Độ tin cậy phân tích
        }
        
        self.logger.info(f"Advanced query optimization: {intent.value} (complexity: {result['query_complexity']})")
        self.logger.info(f"Vietnamese translation: '{query}' → '{translated_query}'")
        return result


class QueryRewriter:
    """
    Advanced query rewriting using templates and synonyms
    """
    
    # Synonym mappings
    SYNONYMS = {
        "ml": "machine learning",
        "ai": "artificial intelligence",
        "rag": "retrieval augmented generation",
        "llm": "large language model",
        "nlp": "natural language processing",
        "cv": "computer vision",
        "gpt": "generative pre-trained transformer"
    }
    
    # Query templates for different intents
    TEMPLATES = {
        "definition": "What is {entity}? Definition and explanation",
        "how_to": "How to {entity}? Step by step guide",
        "reason": "Why is {entity} important? Reasons and benefits",
        "comparison": "Compare {entity1} vs {entity2}. Differences and similarities"
    }
    
    @classmethod
    def expand_abbreviations(cls, query: str) -> str:
        """
        Expand abbreviations in query
        
        Args:
            query: Input query
            
        Returns:
            Query with expanded abbreviations
        """
        result = query
        for abbrev, full in cls.SYNONYMS.items():
            # Case-insensitive replacement with word boundaries
            pattern = r'\b' + abbrev + r'\b'
            result = re.sub(pattern, full, result, flags=re.IGNORECASE)
        
        return result
    
    @classmethod
    def apply_template(cls, template_type: str, entity: str) -> str:
        """
        Apply template to query
        
        Args:
            template_type: Template type (definition, how_to, etc)
            entity: Main entity in query
            
        Returns:
            Rewritten query
        """
        template = cls.TEMPLATES.get(template_type, cls.TEMPLATES["definition"])
        return template.format(entity=entity)

"""
Retrieval Optimizer - Xử lý các vấn đề chính trong truy xuất tài liệu
Đảm bảo tài liệu được truy xuất hoàn toàn liên quan đến câu hỏi của người dùng
"""
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from utils.logger import setup_logger
from utils.text_processor import clean_text, normalize_vietnamese_text, extract_keywords

logger = setup_logger(__name__)


@dataclass
class OptimizedResult:
    """Kết quả tài liệu được tối ưu hóa"""
    content: str
    source: str
    relevance_score: float  # 0-1, dựa trên câu hỏi
    query_match_score: float  # Độ khớp trực tiếp với câu hỏi
    semantic_score: float  # Điểm ngữ nghĩa
    quality_score: float  # Chất lượng tài liệu
    title: str = ""
    url: str = ""
    metadata: Dict = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}
        # Final score = weighted average
        self.final_score = (
            self.query_match_score * 0.35 +
            self.semantic_score * 0.35 +
            self.quality_score * 0.20 +
            self.relevance_score * 0.10
        )


class QueryAnalyzer:
    """Phân tích câu hỏi để tìm từ khóa chính"""
    
    # Các từ dừng thường gặp (stopwords)
    VIETNAMESE_STOPWORDS = {
        'là', 'cái', 'chiếc', 'những', 'được', 'được', 'có', 'có', 'là',
        'những', 'như', 'sao', 'với', 'và', 'hoặc', 'nếu', 'thì', 'để',
        'từ', 'trong', 'trên', 'dưới', 'qua', 'giữa', 'duyên', 'dữ', 'dĩa',
        'đó', 'đây', 'này', 'kia', 'nào', 'ai', 'gì', 'ai', 'em', 'tôi',
        'bạn', 'anh', 'chị', 'các', 'cả', 'toàn', 'tất', 'mà', 'mới', 'bao'
    }
    
    @classmethod
    def extract_keywords(cls, query: str, top_n: int = 5) -> List[str]:
        """
        Trích xuất từ khóa chính từ câu hỏi
        
        Args:
            query: Câu hỏi ban đầu
            top_n: Số từ khóa cần trích
            
        Returns:
            Danh sách từ khóa, sắp xếp theo độ quan trọng
        """
        # Làm sạch và tách từ
        query_clean = clean_text(query).lower()
        words = query_clean.split()
        
        # Loại bỏ stopwords và lấy từ có độ dài >= 3
        keywords = [
            w for w in words 
            if w not in cls.VIETNAMESE_STOPWORDS and len(w) >= 3
        ]
        
        # Loại bỏ trùng lặp, giữ thứ tự
        seen = set()
        unique_keywords = []
        for w in keywords:
            if w not in seen:
                unique_keywords.append(w)
                seen.add(w)
        
        logger.info(f"Extracted keywords from '{query}': {unique_keywords[:top_n]}")
        return unique_keywords[:top_n]
    
    @classmethod
    def analyze_intent(cls, query: str) -> Dict[str, any]:
        """
        Phân tích ý định của câu hỏi
        - Enhanced decision making cho các loại câu hỏi
        - Đặc biệt ưu tiên câu hỏi định nghĩa
        
        Args:
            query: Câu hỏi
            
        Returns:
            Dict với intent_type, keywords, complexity
        """
        query_lower = query.lower()
        
        # Phát hiện loại câu hỏi với patterns cũng
        if query_lower.startswith(("là gì", "cái gì", "định nghĩa", "khái niệm")) or \
           any(p in query_lower for p in ["là gì", "what is", "what's", "define", "definition"]):
            intent_type = "definition"
        elif query_lower.startswith(("làm sao", "cách", "như thế nào", "phương pháp")) or \
             any(p in query_lower for p in ["how to", "how do", "how can"]):
            intent_type = "how-to"
        elif query_lower.startswith(("tại sao", "vì sao")) or \
             any(p in query_lower for p in ["why", "why is"]):
            intent_type = "why"
        elif query_lower.startswith(("so sánh", "khác nhau", "giống nhau")) or \
             any(p in query_lower for p in ["vs", "versus", "compare"]):
            intent_type = "comparison"
        elif query_lower.startswith(("có", "liệu")) or \
             any(p in query_lower for p in ["is", "are", "does"]):
            intent_type = "yes-no"
        else:
            intent_type = "general"
        
        keywords = cls.extract_keywords(query)
        
        return {
            "intent_type": intent_type,
            "keywords": keywords,
            "complexity": min(len(keywords), 3),  # 1-3
            "is_detailed_question": len(query) > 50
        }


class RelevanceEvaluator:
    """Đánh giá độ liên quan của tài liệu với câu hỏi"""
    
    def __init__(self):
        """Khởi tạo evaluator"""
        self.query_analyzer = QueryAnalyzer()
        self.vectorizer = None
        self.query_vector = None
        
    def calculate_query_match_score(self, 
                                    document: str, 
                                    query: str,
                                    keywords: List[str]) -> float:
        """
        Tính điểm khớp trực tiếp giữa document và câu hỏi - ENHANCED
        - Kiểm tra xem keyword chính có xuất hiện trong document
        - Xem xét vị trí của keyword (đầu, giữa, cuối)
        - Phân tích mật độ từ khóa (keyword density)
        - Kiểm tra các cụm từ liên quan
        
        Args:
            document: Nội dung tài liệu
            query: Câu hỏi
            keywords: Danh sách từ khóa
            
        Returns:
            Điểm từ 0-1
        """
        doc_lower = document.lower()
        query_lower = query.lower()
        
        # Kiểm tra toàn bộ query có trong document không
        if query_lower in doc_lower:
            return 0.95
        
        if not keywords or len(keywords) == 0:
            return 0.3
        
        # Đếm số keywords xuất hiện
        keyword_count = sum(1 for kw in keywords if kw in doc_lower)
        keyword_ratio = keyword_count / len(keywords)
        
        # Tính mật độ từ khóa (keyword density)
        total_words = len(doc_lower.split())
        keyword_density = keyword_count / max(total_words, 1)
        
        # Bonus nếu keywords ở đầu document (signaling importance)
        position_bonus = 0
        doc_sents = [s.strip() for s in doc_lower.split('.') if s.strip()]
        if doc_sents:
            first_sent = doc_sents[0].lower()
            for i, kw in enumerate(keywords[:3]):  # Top 3 keywords
                if kw in first_sent:
                    position_bonus += (0.15 - i * 0.04)  # Decreasing bonus
        
        # Bonus nếu keywords xuất hiện gần nhau (cohesion)
        cohesion_bonus = 0
        if len(keywords) > 1:
            for i, kw1 in enumerate(keywords[:2]):
                for kw2 in keywords[i+1:3]:
                    # Find both in same sentence
                    for sent in doc_sents:
                        if kw1 in sent and kw2 in sent:
                            cohesion_bonus += 0.1
        
        # Tính điểm cuối cùng
        score = (
            keyword_ratio * 0.5 +  # 50% dựa trên số lượng keywords
            min(keyword_density * 10, 0.2) * 0.25 +  # 25% dựa trên mật độ
            position_bonus * 0.15 +  # 15% dựa trên vị trí
            cohesion_bonus * 0.10  # 10% dựa trên sự kết hợp
        )
        
        return min(score, 1.0)
    
    def calculate_semantic_score(self, 
                                document: str, 
                                query: str) -> float:
        """
        Tính điểm ngữ nghĩa bằng cosine similarity - ENHANCED
        - Sử dụng TF-IDF cho weighted similarity
        - Fallback sử dụng overlap semantic
        - Xử lý tiếng Việt tốt hơn
        
        Args:
            document: Nội dung tài liệu
            query: Câu hỏi
            
        Returns:
            Điểm từ 0-1
        """
        try:
            # Làm sạch văn bản
            query_clean = clean_text(query).lower()
            doc_clean = clean_text(document).lower()
            
            # Loại bỏ stopwords thông dụng
            common_stopwords = {
                'là', 'một', 'các', 'được', 'này', 'đó', 'với', 'và', 'hoặc',
                'để', 'từ', 'trong', 'trên', 'dưới', 'có', 'có', 'những',
                'the', 'a', 'an', 'and', 'or', 'is', 'are', 'be', 'been'
            }
            
            query_words = set(w for w in query_clean.split() 
                            if w not in common_stopwords and len(w) > 2)
            doc_words = set(w for w in doc_clean.split() 
                          if w not in common_stopwords and len(w) > 2)
            
            if not query_words or not doc_words:
                return 0.3
            
            # Tính overlap
            overlap = len(query_words & doc_words)
            union = len(query_words | doc_words)
            
            jaccard_similarity = overlap / union if union > 0 else 0
            
            # Tính document relevance (mức độ dài kết hợp)
            max_len = max(len(query_words), len(doc_words))
            recall = overlap / max_len if max_len > 0 else 0
            
            # Kết hợp: Jaccard 60% + Recall 40%
            semantic_score = jaccard_similarity * 0.6 + recall * 0.4
            
            # Thêm bonus nếu document dài (chứa thêm thông tin)
            if len(doc_clean) > len(query_clean) * 3:
                semantic_score = min(semantic_score + 0.1, 1.0)
            
            return float(semantic_score)
            
        except Exception as e:
            logger.debug(f"Error in semantic scoring: {e}")
            return 0.3
    
    def calculate_quality_score(self, 
                               document: str,
                               title: str = "",
                               url: str = "") -> float:
        """
        Tính điểm chất lượng tài liệu - ENHANCED
        - Độ dài hợp lý (> 50 ký tự)
        - Có tiêu đề mô tả thích hợp
        - Có URL đáng tin cậy
        - Không có keyword spam
        - Độ chi tiết (số câu, số từ)
        - Cấu trúc và định dạng
        
        Args:
            document: Nội dung
            title: Tiêu đề
            url: URL nguồn
            
        Returns:
            Điểm từ 0-1
        """
        score = 0.3  # Base score (higher starting point)
        
        # 1. Kiểm tra độ dài
        doc_len = len(document)
        if doc_len > 500:
            score += 0.25
        elif doc_len > 200:
            score += 0.18
        elif doc_len > 100:
            score += 0.10
        elif doc_len > 50:
            score += 0.05
        
        # 2. Kiểm tra tiêu đề
        if title and len(title) > 5:
            title_quality = min(len(title) / 100, 1.0)  # Better longer titles
            score += 0.15 * title_quality
        
        # 3. Kiểm tra URL độ tin cậy
        if url:
            if url.startswith(('https://', 'http://')):
                score += 0.10
            
            # Bonus cho các domain đáng tin cậy
            trusted_domains = [
                'wikipedia', 'github', 'stackoverflow', '.edu', '.gov',
                'scholar', 'researchgate', 'arxiv'
            ]
            for domain in trusted_domains:
                if domain in url.lower():
                    score += 0.05
                    break
        
        # 4. Kiểm tra độ chi tiết (số câu, số từ)
        sentences = [s.strip() for s in document.split('.') if s.strip()]
        total_words = len(document.split())
        
        if len(sentences) > 5:  # Multiple sentences = more detailed
            score += 0.10
        
        if total_words > 100:  # Good keyword coverage
            avg_sentence_len = total_words / len(sentences) if sentences else 0
            if 10 < avg_sentence_len < 30:  # Good sentence length
                score += 0.05
        
        # 5. Kiểm tra keyword spam (từ lặp quá nhiều)
        words = document.lower().split()
        if len(words) > 0:
            word_freq = {}
            for w in words:
                if len(w) > 3:  # Chỉ tính từ dài
                    word_freq[w] = word_freq.get(w, 0) + 1
            
            if word_freq:
                max_freq = max(word_freq.values())
                spam_ratio = max_freq / len(words)
                
                # Penalize high frequency words
                if spam_ratio > 0.4:  # Một từ chiếm > 40%
                    score -= 0.25
                elif spam_ratio > 0.3:
                    score -= 0.15
                elif spam_ratio > 0.2:
                    score -= 0.05
        
        # 6. Kiểm tra formatting (lists, emphasis, etc.)
        format_indicators = [
            ('•', 0.05),
            ('-', 0.03),
            ('**', 0.05),  # Bold
            ('__', 0.05),  # Underline
            ('`', 0.03),   # Code
        ]
        
        for indicator, bonus in format_indicators:
            if indicator in document:
                score += bonus * min(document.count(indicator) / 10, 1)
        
        # 7. Kiểm tra ngôn ngữ (basic language quality)
        if any(c.isdigit() for c in document):  # Has numbers/data
            score += 0.05
        
        if re.search(r'[A-Z][a-z]+', document):  # Proper capitalization
            score += 0.03
        
        return min(max(score, 0), 1.0)


class DocumentFilter:
    """Lọc và loại bỏ tài liệu không liên quan"""
    
    @staticmethod
    def has_relevant_content(document: str, keywords: List[str], threshold: float = 0.5) -> bool:
        """
        Kiểm tra xem tài liệu có chứa nội dung liên quan không
        
        Args:
            document: Nội dung tài liệu
            keywords: Danh sách từ khóa
            threshold: Ngưỡng (0-1)
            
        Returns:
            True nếu liên quan, False nếu không
        """
        if not keywords:
            return True
        
        doc_lower = document.lower()
        keyword_matches = sum(1 for kw in keywords if kw in doc_lower)
        match_ratio = keyword_matches / len(keywords)
        
        return match_ratio >= threshold
    
    @staticmethod
    def is_spam_document(document: str) -> bool:
        """
        Phát hiện xem tài liệu có phải spam không
        
        Args:
            document: Nội dung
            
        Returns:
            True nếu là spam
        """
        doc_lower = document.lower()
        
        # Kiểm tra các pattern spam
        spam_patterns = [
            r'(click here|buy now|subscribe|follow us)',  # Spam keywords
            r'(\b\w+\b){1}(\s+\1){5,}',  # Từ lặp quá nhiều
            r'^[\W_]{10,}$',  # Chỉ ký tự đặc biệt
            r'(viagra|casino|poker|xxx)',  # Adult spam
        ]
        
        for pattern in spam_patterns:
            if re.search(pattern, doc_lower):
                return True
        
        return False
    
    @staticmethod
    def remove_duplicates(documents: List[Dict], similarity_threshold: float = 0.9) -> List[Dict]:
        """
        Loại bỏ các tài liệu trùng lặp hoặc rất giống nhau
        
        Args:
            documents: Danh sách tài liệu
            similarity_threshold: Ngưỡng tương tự (0-1)
            
        Returns:
            Danh sách tài liệu không trùng
        """
        if len(documents) <= 1:
            return documents
        
        unique_docs = []
        
        for doc in documents:
            is_duplicate = False
            doc_content = doc.get('content', '').lower()
            
            for unique_doc in unique_docs:
                unique_content = unique_doc.get('content', '').lower()
                
                # So sánh: nếu một là substring của cái kia
                if (len(doc_content) > 30 and len(unique_content) > 30):
                    if doc_content in unique_content or unique_content in doc_content:
                        is_duplicate = True
                        break
                    
                    # Hoặc check hash similarity
                    overlap = len(set(doc_content.split()) & set(unique_content.split()))
                    total = len(set(doc_content.split()) | set(unique_content.split()))
                    if total > 0 and overlap / total > similarity_threshold:
                        is_duplicate = True
                        break
            
            if not is_duplicate:
                unique_docs.append(doc)
        
        logger.info(f"Removed {len(documents) - len(unique_docs)} duplicate documents")
        return unique_docs


class RetrievelOptimizer:
    """
    Trình tối ưu hóa truy xuất chính
    Kết hợp tất cả các thành phần để cải thiện chất lượng tài liệu được truy xuất
    """
    
    def __init__(self):
        """Khởi tạo optimizer"""
        self.query_analyzer = QueryAnalyzer()
        self.relevance_evaluator = RelevanceEvaluator()
        self.document_filter = DocumentFilter()
        logger.info("Retrieval Optimizer initialized")
    
    def optimize_results(self, 
                        documents: List[Dict],
                        query: str,
                        top_k: int = 5,
                        min_relevance_score: float = 0.3) -> List[OptimizedResult]:
        """
        Tối ưu hóa danh sách tài liệu được truy xuất - ENHANCED
        - Xử lý riêng cho câu hỏi định nghĩa
        - Ưu tiên Wikipedia, GitHub, tài liệu chính thức
        - Loại bỏ khóa học, quảng cáo, forum spam
        
        Các bước:
        1. Phân tích câu hỏi để lấy keywords và ý định
        2. Lọc spam, quảng cáo, khóa học, forum spam
        3. Loại bỏ duplicates
        4. Tính điểm liên quan cho mỗi tài liệu
        5. Sắp xếp theo điểm (với tính đến ý định câu hỏi)
        6. Áp dụng ngưỡng động
        7. Trả về top-k
        
        Args:
            documents: Danh sách tài liệu thô
            query: Câu hỏi ban đầu
            top_k: Số tài liệu kết quả
            min_relevance_score: Ngưỡng điểm tối thiểu
            
        Returns:
            Danh sách tài liệu tối ưu
        """
        logger.info(f"Optimizing {len(documents)} documents for query: '{query}' (top_k={top_k})")
        
        if not documents:
            logger.warning("No documents to optimize")
            return []
        
        # Bước 1: Phân tích câu hỏi
        query_analysis = self.query_analyzer.analyze_intent(query)
        keywords = query_analysis['keywords']
        query_complexity = query_analysis['complexity']
        intent_type = query_analysis['intent_type']
        
        logger.info(f"Query analysis: intent={intent_type}, keywords={keywords}, complexity={query_complexity}")
        
        # Tính ngưỡng động dựa trên độ phức tạp và loại câu hỏi
        dynamic_threshold = min_relevance_score - (query_complexity * 0.05)
        
        # Cho câu hỏi định nghĩa, tăng threshold (chỉ lấy kết quả chất lượng cao)
        if intent_type == "definition":
            dynamic_threshold = min_relevance_score + 0.1
            logger.info(f"Definition query detected - increased threshold to {dynamic_threshold:.2f}")
        
        dynamic_threshold = max(dynamic_threshold, 0.15)  # Không quá thấp
        
        # Bước 2: Lọc spam, quảng cáo, khóa học, forum
        filtered_docs = self._filter_documents(documents, keywords, intent_type)
        
        # Loại bỏ duplicates
        filtered_docs = self.document_filter.remove_duplicates(filtered_docs)
        logger.info(f"After filtering: {len(filtered_docs)} documents remain")
        
        # Bước 3: Tính điểm cho mỗi tài liệu
        optimized_results = self._score_documents(
            filtered_docs, query, keywords, query_complexity, intent_type
        )
        
        # Bước 4: Sắp xếp theo điểm giảm dần
        optimized_results.sort(key=lambda x: x.final_score, reverse=True)
        
        # Bước 5: Trả về top-k
        final_results = optimized_results[:top_k]
        
        logger.info(f"Optimization complete: {len(final_results)} documents selected (threshold={dynamic_threshold:.2f})")
        for i, doc in enumerate(final_results, 1):
            logger.info(f"  {i}. Score={doc.final_score:.3f} "
                       f"| Match={doc.query_match_score:.2f} "
                       f"| Semantic={doc.semantic_score:.2f} "
                       f"| Quality={doc.quality_score:.2f} "
                       f"| {doc.title[:40] if doc.title else doc.source}")
        
        return final_results
    
    def _filter_documents(self, documents: List[Dict], keywords: List[str], intent_type: str) -> List[Dict]:
        """
        Lọc tài liệu không liên quan, spam, quảng cáo, khóa học
        
        Args:
            documents: Danh sách tài liệu thô
            keywords: Danh sách keywords
            intent_type: Loại câu hỏi (definition, how-to, etc.)
            
        Returns:
            Danh sách tài liệu đã lọc
        """
        filtered = []
        threshold_adjust = 0.3 if len(keywords) < 2 else 0.4
        
        # Patterns cần loại bỏ
        spam_patterns = [
            (r'(udemy|coursera|skillshare|pluralsight)', 'course platform'),
            (r'(enroll|register.*course|take.*course|buy.*course|purchase.*course)', 'course content'),
            (r'(click here|buy now|subscribe|sign up)', 'spam CTA'),
            (r'(sponsored|advertisement|ad by)', 'advertisement'),
            (r'(viagra|casino|poker|lottery)', 'adult spam'),
        ]
        
        # Forum spam patterns (cho câu hỏi định nghĩa)
        forum_spam = r'(forum|qa|answers|discussions|reddit.*answered)' if intent_type == "definition" else None
        
        for doc in documents:
            content = (doc.get('content', '') + ' ' + doc.get('title', '')).lower()
            url = doc.get('url', '').lower()
            
            # Kiểm tra spam
            is_spam = False
            for pattern, spam_type in spam_patterns:
                if re.search(pattern, content, re.IGNORECASE) or re.search(pattern, url, re.IGNORECASE):
                    logger.debug(f"Removed {spam_type}: {doc.get('title', 'Unknown')[:30]}")
                    is_spam = True
                    break
            
            if is_spam:
                continue
            
            # Forum spam filtering (cho definition queries)
            if forum_spam and re.search(forum_spam, url, re.IGNORECASE):
                logger.debug(f"Removed forum content: {doc.get('title', 'Unknown')[:30]}")
                continue
            
            # Kiểm tra liên quan
            if not self.document_filter.has_relevant_content(
                content, keywords, threshold=threshold_adjust):
                logger.debug(f"Removed irrelevant: {doc.get('title', 'Unknown')[:30]}")
                continue
            
            filtered.append(doc)
        
        return filtered
    
    def _score_documents(self, documents: List[Dict], query: str, keywords: List[str],
                        query_complexity: int, intent_type: str) -> List[OptimizedResult]:
        """
        Tính điểm cho tất cả tài liệu
        
        Args:
            documents: Danh sách tài liệu đã lọc
            query: Câu hỏi ban đầu
            keywords: Danh sách keywords
            query_complexity: Độ phức tạp câu hỏi
            intent_type: Loại câu hỏi
            
        Returns:
            Danh sách OptimizedResult
        """
        optimized_results = []
        
        for doc in documents:
            content = doc.get('content', '')
            title = doc.get('title', '')
            url = doc.get('url', '')
            source = doc.get('source', 'unknown')
            initial_score = doc.get('score', 0.5)
            
            # Tính các điểm cải tiến
            query_match = self.relevance_evaluator.calculate_query_match_score(
                content, query, keywords)
            semantic = self.relevance_evaluator.calculate_semantic_score(content, query)
            quality = self.relevance_evaluator.calculate_quality_score(content, title, url)
            
            # Điều chỉnh trọng số dựa trên loại câu hỏi
            if intent_type == "definition":
                # Cho định nghĩa: ưu tiên quality + semantic > direct match
                weights = (0.25, 0.40, 0.35)  # query_match, semantic, quality
                
                # Bonus cho Wikipedia, GitHub, tài liệu chính thức
                if 'wikipedia' in url.lower():
                    quality = min(1.0, quality + 0.2)
                elif any(d in url.lower() for d in ['github', '.edu', '.gov', 'documentation']):
                    quality = min(1.0, quality + 0.1)
                
            elif query_complexity >= 2:  # Complex query
                weights = (0.30, 0.45, 0.25)  # query_match, semantic, quality
            else:  # Simple query
                weights = (0.40, 0.35, 0.25)
            
            # Tạo kết quả tối ưu
            result = OptimizedResult(
                content=content,
                source=source,
                relevance_score=initial_score,
                query_match_score=query_match,
                semantic_score=semantic,
                quality_score=quality,
                title=title,
                url=url,
                metadata={
                    'query_intent': intent_type,
                    'keyword_matches': sum(1 for kw in keywords if kw in content.lower()),
                    'semantic_similarity': semantic,
                    'match_quality': query_match
                }
            )
            
            # Recalculate final_score with adjusted weights
            result.final_score = (
                query_match * weights[0] +
                semantic * weights[1] +
                quality * weights[2]
            )
            
            # Ngưỡng động cho định nghĩa (cao hơn)
            min_threshold = 0.45 if intent_type == "definition" else 0.3
            
            if result.final_score >= min_threshold:
                optimized_results.append(result)
        
        return optimized_results
    
    def get_recommendations(self, optimized_results: List[OptimizedResult]) -> Dict:
        """
        Cung cấp khuyến nghị cải thiện dựa trên kết quả tối ưu
        
        Args:
            optimized_results: Danh sách kết quả tối ưu
            
        Returns:
            Dict với các khuyến nghị
        """
        if not optimized_results:
            return {
                "status": "warning",
                "message": "Không tìm thấy tài liệu liên quan",
                "recommendation": "Hãy thử đặt câu hỏi khác hoặc cung cấp thêm ngữ cảnh"
            }
        
        avg_semantic = np.mean([r.semantic_score for r in optimized_results])
        avg_query_match = np.mean([r.query_match_score for r in optimized_results])
        avg_quality = np.mean([r.quality_score for r in optimized_results])
        
        recommendations = []
        
        if avg_semantic < 0.5:
            recommendations.append("Chất lượng ngữ nghĩa thấp - Câu hỏi có thể không rõ ràng")
        
        if avg_query_match < 0.5:
            recommendations.append("Độ khớp từ khóa thấp - Tài liệu có thể bàn về chủ đề liên quan")
        
        if avg_quality < 0.5:
            recommendations.append("Chất lượng tài liệu thấp - Nên xem xét các nguồn khác")
        
        return {
            "status": "success",
            "avg_scores": {
                "semantic": float(avg_semantic),
                "query_match": float(avg_query_match),
                "quality": float(avg_quality)
            },
            "total_documents": len(optimized_results),
            "recommendations": recommendations if recommendations else ["Kết quả tốt!"]
        }

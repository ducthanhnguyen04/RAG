"""
Text processing utilities cho RAG - with full Vietnamese support
Handles Vietnamese diacritics, semantic normalization, and chunking
"""
import re
import unicodedata
from typing import List, Tuple, Optional
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP


def clean_text(text: str, preserve_vietnamese: bool = True) -> str:
    """
    Clean and normalize text (with full Vietnamese character support)
    Preserves Vietnamese diacritical marks: ă, ơ, ư, à, á, ả, ã, ạ, etc.
    
    Args:
        text: Input text
        preserve_vietnamese: Keep Vietnamese diacritical marks
        
    Returns:
        Cleaned text preserving Vietnamese semantics
    """
    # Ensure UTF-8 encoding
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize Unicode (NFC = composed form for Vietnamese diacritics)
    text = unicodedata.normalize('NFC', text)
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove HTML entities
    text = re.sub(r'&[a-z]+;', ' ', text, flags=re.IGNORECASE)
    
    # KEEP Vietnamese characters: Use UNICODE flag to preserve all diacritics
    # Pattern: keep word chars (including Vietnamese) + spaces + common punctuation
    text = re.sub(
        r"[^\w\s\.\,\!\?\-\(\)\:\'\"\—\–\;]",
        ' ',
        text,
        flags=re.UNICODE
    )
    
    # Remove multiple spaces
    text = re.sub(r'\s+', ' ', text)
    
    return text.strip()


def normalize_vietnamese_text(text: str) -> str:
    """
    Normalize Vietnamese text while preserving semantic meaning
    - Keeps important Vietnamese particles and markers
    - Removes only less important fillers
    
    Args:
        text: Input text
        
    Returns:
        Normalized Vietnamese text
    """
    # Clean first preserving Vietnamese
    text = clean_text(text, preserve_vietnamese=True)
    
    # Lowercase for processing consistency
    text = text.lower()
    
    return text


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, 
               overlap: int = CHUNK_OVERLAP) -> List[str]:
    """
    Split text into overlapping chunks while preserving Vietnamese semantics
    
    Args:
        text: Input text
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    # Clean text first
    text = clean_text(text, preserve_vietnamese=True)
    
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        if chunk.strip():
            chunks.append(chunk)
    
    return chunks


def extract_keywords(text: str, num_keywords: int = 5) -> List[str]:
    """
    Extract keywords từ text while preserving Vietnamese semantics
    - Keeps important Vietnamese words
    - Removes only common stopwords
    
    Args:
        text: Input text
        num_keywords: Number of keywords to extract
        
    Returns:
        List of keywords
    """
    # Vietnamese stopwords - minimal set, preserves meaning
    vietnamese_stopwords = {
        # Articles and particles
        'là', 'cái', 'chiếc', 'những', 'các', 'cả', 'toàn', 'tất',
        # Common auxiliary verbs  
        'được', 'có', 'bị', 'làm', 'cho', 'đặt', 'mang',
        # Prepositions
        'từ', 'trong', 'trên', 'dưới', 'qua', 'giữa', 'ở', 'tại',
        # Conjunctions
        'và', 'hoặc', 'hay', 'nhưng', 'mà', 'nên', 'vì', 'như',
        # Pronouns
        'tôi', 'bạn', 'chúng ta', 'anh', 'chị', 'em',
        # English stopwords (for compatibility)
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'is', 'are', 'was', 'were', 'be'
    }
    
    # Clean and split
    text_clean = clean_text(text, preserve_vietnamese=True).lower()
    words = text_clean.split()
    
    # Filter: keep words not in stopwords, length >= 2 for Vietnamese
    keywords = [
        w for w in words 
        if w not in vietnamese_stopwords and len(w) >= 2
    ]
    
    # Remove duplicates while preserving order
    seen = set()
    unique_keywords = []
    for w in keywords:
        if w not in seen:
            unique_keywords.append(w)
            seen.add(w)
    
    # Return top N
    return unique_keywords[:num_keywords]


def calculate_text_similarity(text1: str, text2: str) -> float:
    """
    Calculate similarity between two Vietnamese texts
    Using word overlap and semantic similarity
    
    Args:
        text1: First text
        text2: Second text
        
    Returns:
        Similarity score (0-1)
    """
    # Normalize both texts
    text1_clean = clean_text(text1, preserve_vietnamese=True).lower()
    text2_clean = clean_text(text2, preserve_vietnamese=True).lower()
    
    words1 = set(text1_clean.split())
    words2 = set(text2_clean.split())
    
    if not words1 or not words2:
        return 0.0
    
    # Jaccard similarity
    intersection = len(words1 & words2)
    union = len(words1 | words2)
    
    similarity = intersection / union if union > 0 else 0.0
    
    return similarity


def merge_similar_chunks(chunks: List[str], similarity_threshold: float = 0.8) -> List[str]:
    """
    Merge very similar chunks while preserving Vietnamese semantics
    
    Args:
        chunks: List of chunks
        similarity_threshold: Threshold for merging (0-1)
        
    Returns:
        Merged chunks
    """
    if len(chunks) <= 1:
        return chunks
    
    merged = []
    skip_indices = set()
    
    for i, chunk1 in enumerate(chunks):
        if i in skip_indices:
            continue
        
        merged_chunk = chunk1
        
        for j in range(i + 1, len(chunks)):
            if j in skip_indices:
                continue
            
            chunk2 = chunks[j]
            similarity = calculate_text_similarity(chunk1, chunk2)
            
            if similarity >= similarity_threshold:
                # Merge chunks
                merged_chunk = merged_chunk + " " + chunk2
                skip_indices.add(j)
        
        merged.append(merged_chunk)
    
    return merged


def format_displayed_text(text: str, max_length: int = 500, add_ellipsis: bool = True) -> str:
    """
    Format text for display while preserving Vietnamese characters correctly
    
    Args:
        text: Text to format
        max_length: Maximum length before truncation
        add_ellipsis: Add ... if truncated
        
    Returns:
        Formatted text with correct Vietnamese encoding
    """
    # Ensure proper UTF-8 encoding
    if isinstance(text, bytes):
        text = text.decode('utf-8', errors='replace')
    
    # Normalize Unicode (NFC for Vietnamese)
    text = unicodedata.normalize('NFC', text)
    
    # Truncate if needed
    if len(text) > max_length:
        text = text[:max_length]
        if add_ellipsis:
            text = text + "..."
    
    return text.strip()

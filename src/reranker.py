"""
Advanced Reranking System
Supports multiple reranking strategies:
1. Cross-Encoder: Transformer-based relevance scoring
2. LLM-based: Using OpenAI API for intelligent ranking
3. Semantic Cohesion: Measure semantic flow between documents
"""
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
import numpy as np
import os
from utils.logger import setup_logger

# Disable CUDA to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

logger = setup_logger(__name__)


@dataclass
class RankedDocument:
    """Document with reranking score"""
    content: str
    source: str
    original_score: float = 0.0
    rerank_score: float = 0.0
    final_score: float = 0.0
    reasoning: str = ""


class CrossEncoderReranker:
    """
    Cross-Encoder based Reranker
    Uses sentence-transformers cross-encoder for precise relevance ranking
    """
    
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        """
        Initialize Cross-Encoder Reranker
        
        Args:
            model_name: Cross-encoder model name
        """
        try:
            from sentence_transformers import CrossEncoder
            self.cross_encoder = CrossEncoder(model_name)
            self.model_name = model_name
            logger.info(f"✓ Cross-Encoder Reranker initialized: {model_name}")
        except ImportError:
            logger.warning("sentence-transformers not found. Install: pip install sentence-transformers")
            self.cross_encoder = None
    
    def rerank(self, 
               query: str, 
               documents: List[Dict],
               original_scores: Optional[List[float]] = None,
               top_k: Optional[int] = None) -> List[RankedDocument]:
        """
        Rerank documents using cross-encoder
        
        Args:
            query: User query
            documents: List of documents with 'content' and optional 'source'
            original_scores: Original scores from retrieval (for comparison)
            top_k: Return only top-k documents
            
        Returns:
            List of RankedDocument with rerank scores
        """
        if not self.cross_encoder:
            logger.error("Cross-encoder not available")
            return self._fallback_rerank(query, documents, original_scores, top_k)
        
        if not documents:
            return []
        
        # Prepare document texts
        doc_texts = [d.get("content", str(d)) if isinstance(d, dict) else str(d) 
                     for d in documents]
        doc_sources = [d.get("source", "Unknown") if isinstance(d, dict) else "Unknown" 
                       for d in documents]
        
        try:
            # Generate query-document pairs and score them
            pairs = [[query, doc] for doc in doc_texts]
            scores = self.cross_encoder.predict(pairs)
            
            # Normalize scores to 0-1
            scores_normalized = self._normalize_scores(scores)
            
            # Create ranked documents
            ranked_docs = []
            for i, (doc_text, source, score) in enumerate(zip(doc_texts, doc_sources, scores_normalized)):
                original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
                
                ranked_doc = RankedDocument(
                    content=doc_text,
                    source=source,
                    original_score=original_score,
                    rerank_score=float(score),
                    final_score=float(score) * 0.85 + original_score * 0.15,  # Stronger reranking weight
                    reasoning=f"Cross-Encoder relevance: {score:.3f}"
                )
                ranked_docs.append(ranked_doc)
            
            # Sort by final score descending
            ranked_docs.sort(key=lambda x: x.final_score, reverse=True)
            
            # Apply top-k if specified
            if top_k:
                ranked_docs = ranked_docs[:top_k]
            
            logger.info(f"Reranked {len(doc_texts)} documents using cross-encoder")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Cross-encoder reranking error: {e}")
            return self._fallback_rerank(query, documents, original_scores, top_k)
    
    def _normalize_scores(self, scores):
        """Normalize scores to 0-1 range"""
        scores = np.array(scores)
        min_score = scores.min()
        max_score = scores.max()
        
        if max_score == min_score:
            return (scores - min_score) / (max_score - min_score + 1e-10)
        
        return (scores - min_score) / (max_score - min_score)
    
    def _fallback_rerank(self, query, documents, original_scores=None, top_k=None):
        """Fallback reranking if cross-encoder fails"""
        ranked_docs = []
        for i, doc in enumerate(documents):
            doc_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
            source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
            original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
            
            ranked_docs.append(RankedDocument(
                content=doc_text,
                source=source,
                original_score=original_score,
                rerank_score=original_score,
                final_score=original_score,
                reasoning="Fallback (cross-encoder unavailable)"
            ))
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        return ranked_docs


class LLMReranker:
    """
    LLM-based Reranker using OpenAI
    Uses LLM intelligence to rank documents by relevance
    """
    
    def __init__(self, api_key: str = None, model: str = "gpt-3.5-turbo"):
        """
        Initialize LLM Reranker
        
        Args:
            api_key: OpenAI API key
            model: Model name
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(api_key=api_key)
            self.model = model
            self.api_key = api_key
            logger.info(f"✓ LLM Reranker initialized: {model}")
        except ImportError:
            logger.warning("openai not found. Install: pip install openai")
            self.client = None
    
    def rerank(self, 
               query: str, 
               documents: List[Dict],
               original_scores: Optional[List[float]] = None,
               top_k: Optional[int] = 5) -> List[RankedDocument]:
        """
        Rerank documents using LLM
        
        Args:
            query: User query
            documents: List of documents
            original_scores: Original scores
            top_k: Return top-k documents
            
        Returns:
            List of RankedDocument
        """
        if not self.client:
            logger.error("OpenAI client not available")
            return self._fallback_rerank(query, documents, original_scores, top_k)
        
        if not documents or len(documents) == 0:
            return []
        
        try:
            # Prepare document list
            doc_list = []
            for i, doc in enumerate(documents):
                doc_text = doc.get("content", str(doc))[:200] if isinstance(doc, dict) else str(doc)[:200]
                doc_list.append(f"{i+1}. {doc_text}")
            
            # Create ranking prompt
            prompt = f"""Given the query: "{query}"

Rank these documents by relevance (1=most relevant, {len(documents)}=least relevant).
Return a JSON object like {{"rankings": [1, 3, 2, ...]}}

Documents:
{chr(10).join(doc_list)}

Respond ONLY with valid JSON."""
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a document relevance ranker. Rank documents by relevance to the query."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            # Parse response
            import json
            response_text = response.choices[0].message.content
            ranking_data = json.loads(response_text)
            rankings = ranking_data.get("rankings", list(range(1, len(documents)+1)))
            
            # Convert rankings to scores
            ranked_docs = []
            for i, doc in enumerate(documents):
                doc_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
                source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
                original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
                
                # Rank position (lower is better)
                rank_position = rankings[i] if i < len(rankings) else len(documents)
                rerank_score = 1.0 - (rank_position - 1) / max(len(documents), 1)
                
                ranked_docs.append(RankedDocument(
                    content=doc_text,
                    source=source,
                    original_score=original_score,
                    rerank_score=rerank_score,
                    final_score=rerank_score * 0.85 + original_score * 0.15,
                    reasoning=f"LLM rank: {rank_position}/{len(documents)}"
                ))
            
            # Sort by final score
            ranked_docs.sort(key=lambda x: x.final_score, reverse=True)
            
            if top_k:
                ranked_docs = ranked_docs[:top_k]
            
            logger.info(f"Reranked {len(documents)} documents using LLM")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"LLM reranking error: {e}")
            return self._fallback_rerank(query, documents, original_scores, top_k)
    
    def _fallback_rerank(self, query, documents, original_scores=None, top_k=None):
        """Fallback reranking"""
        ranked_docs = []
        for i, doc in enumerate(documents):
            doc_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
            source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
            original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
            
            ranked_docs.append(RankedDocument(
                content=doc_text,
                source=source,
                original_score=original_score,
                rerank_score=original_score,
                final_score=original_score,
                reasoning="Fallback (LLM unavailable)"
            ))
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        return ranked_docs


class SemanticCoherenceReranker:
    """
    Semantic Coherence-based Reranker
    Measures how well documents flow together semantically around the query
    """
    
    def __init__(self):
        """Initialize coherence reranker"""
        try:
            from sentence_transformers import SentenceTransformer
            self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
            logger.info("✓ Semantic Coherence Reranker initialized")
        except ImportError:
            logger.warning("sentence-transformers not found")
            self.embedder = None
    
    def rerank(self, 
               query: str, 
               documents: List[Dict],
               original_scores: Optional[List[float]] = None,
               top_k: Optional[int] = None) -> List[RankedDocument]:
        """
        Rerank documents by semantic coherence
        
        Args:
            query: User query
            documents: List of documents
            original_scores: Original scores
            top_k: Return top-k documents
            
        Returns:
            List of RankedDocument
        """
        if not self.embedder:
            return self._fallback_rerank(query, documents, original_scores, top_k)
        
        if not documents:
            return []
        
        try:
            # Embed query and documents
            query_embedding = self.embedder.encode(query, convert_to_numpy=True)
            
            doc_texts = [d.get("content", str(d)) if isinstance(d, dict) else str(d) 
                        for d in documents]
            doc_embeddings = self.embedder.encode(doc_texts, convert_to_numpy=True)
            
            # Calculate semantic coherence
            ranked_docs = []
            for i, (doc_text, doc_emb) in enumerate(zip(doc_texts, doc_embeddings)):
                source = documents[i].get("source", "Unknown") if isinstance(documents[i], dict) else "Unknown"
                original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
                
                # Cosine similarity with query
                from sklearn.metrics.pairwise import cosine_similarity
                similarity = cosine_similarity([query_embedding], [doc_emb])[0][0]
                
                # Coherence: how well this document relates to query and other docs
                coherence_score = similarity
                
                ranked_docs.append(RankedDocument(
                    content=doc_text,
                    source=source,
                    original_score=original_score,
                    rerank_score=float(coherence_score),
                    final_score=float(coherence_score) * 0.85 + original_score * 0.15,
                    reasoning=f"Semantic coherence: {coherence_score:.3f}"
                ))
            
            # Sort by final score
            ranked_docs.sort(key=lambda x: x.final_score, reverse=True)
            
            if top_k:
                ranked_docs = ranked_docs[:top_k]
            
            logger.info(f"Reranked {len(doc_texts)} documents by semantic coherence")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Coherence reranking error: {e}")
            return self._fallback_rerank(query, documents, original_scores, top_k)
    
    def _fallback_rerank(self, query, documents, original_scores=None, top_k=None):
        """Fallback reranking"""
        ranked_docs = []
        for i, doc in enumerate(documents):
            doc_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
            source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
            original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
            
            ranked_docs.append(RankedDocument(
                content=doc_text,
                source=source,
                original_score=original_score,
                rerank_score=original_score,
                final_score=original_score,
                reasoning="Fallback (embedder unavailable)"
            ))
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        return ranked_docs


class HybridReranker:
    """
    Hybrid Reranker combining multiple strategies
    Uses weighted combination of different reranking approaches
    """
    
    def __init__(self, 
                 use_cross_encoder: bool = True,
                 use_semantic_coherence: bool = True,
                 use_llm: bool = False,
                 api_key: str = None,
                 weights: Optional[Dict[str, float]] = None):
        """
        Initialize Hybrid Reranker
        
        Args:
            use_cross_encoder: Enable cross-encoder reranker
            use_semantic_coherence: Enable semantic coherence reranker
            use_llm: Enable LLM reranker
            api_key: API key for LLM
            weights: Custom weights for strategies
        """
        self.rerankers = {}
        
        if use_cross_encoder:
            try:
                self.rerankers["cross_encoder"] = CrossEncoderReranker()
            except Exception as e:
                logger.warning(f"Cross-encoder reranker failed: {e}")
        
        if use_semantic_coherence:
            try:
                self.rerankers["semantic"] = SemanticCoherenceReranker()
            except Exception as e:
                logger.warning(f"Semantic coherence reranker failed: {e}")
        
        if use_llm and api_key:
            try:
                self.rerankers["llm"] = LLMReranker(api_key=api_key)
            except Exception as e:
                logger.warning(f"LLM reranker failed: {e}")
        
        # Default weights
        self.weights = weights or {
            "cross_encoder": 0.5,
            "semantic": 0.3,
            "llm": 0.2
        }
        
        # Normalize weights
        total_weight = sum(self.weights.get(k, 0) for k in self.rerankers.keys())
        if total_weight > 0:
            self.weights = {k: v / total_weight for k, v in self.weights.items()}
        
        logger.info(f"✓ Hybrid Reranker initialized with {len(self.rerankers)} strategies")
        logger.info(f"  Weights: {self.weights}")
    
    def rerank(self, 
               query: str, 
               documents: List[Dict],
               original_scores: Optional[List[float]] = None,
               top_k: Optional[int] = None) -> List[RankedDocument]:
        """
        Rerank documents using hybrid approach
        
        Args:
            query: User query
            documents: List of documents
            original_scores: Original scores
            top_k: Return top-k documents
            
        Returns:
            List of RankedDocument
        """
        if not documents:
            return []
        
        if not self.rerankers:
            logger.warning("No rerankers available, returning original order")
            return [RankedDocument(
                content=d.get("content", str(d)),
                source=d.get("source", "Unknown"),
                original_score=original_scores[i] if original_scores and i < len(original_scores) else 0.0,
                rerank_score=0.0,
                final_score=original_scores[i] if original_scores and i < len(original_scores) else 0.0
            ) for i, d in enumerate(documents)]
        
        # Run all rerankers
        all_scores = {name: {} for name in self.rerankers.keys()}
        
        for name, reranker in self.rerankers.items():
            try:
                ranked = reranker.rerank(query, documents, original_scores)
                for i, ranked_doc in enumerate(ranked):
                    all_scores[name][i] = ranked_doc.rerank_score
            except Exception as e:
                logger.error(f"Reranker {name} failed: {e}")
        
        # Combine scores using weights
        final_scores = {}
        for i in range(len(documents)):
            weighted_score = 0.0
            for name, weight in self.weights.items():
                if i in all_scores.get(name, {}):
                    weighted_score += all_scores[name][i] * weight
            final_scores[i] = weighted_score
        
        # Create combined ranked documents
        ranked_docs = []
        for i, doc in enumerate(documents):
            doc_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
            source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
            original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
            rerank_score = final_scores.get(i, 0.0)
            
            ranked_docs.append(RankedDocument(
                content=doc_text,
                source=source,
                original_score=original_score,
                rerank_score=rerank_score,
                final_score=rerank_score * 0.85 + original_score * 0.15,
                reasoning=f"Hybrid score from {len(self.rerankers)} strategies"
            ))
        
        # Sort by final score
        ranked_docs.sort(key=lambda x: x.final_score, reverse=True)
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        
        logger.info(f"Hybrid reranked {len(documents)} documents using {len(self.rerankers)} strategies")
        return ranked_docs


class MockLLMReranker:
    """
    Mock LLM Reranker using built-in Mock LLM
    Evaluates document relevance without needing OpenAI API
    Creates clear score differentiation between relevant and irrelevant documents
    """
    
    def __init__(self):
        """Initialize Mock LLM Reranker"""
        try:
            from src.llm_generator import create_llm_generator
            self.llm = create_llm_generator()
            logger.info("✓ Mock LLM Reranker initialized")
        except Exception as e:
            logger.warning(f"Mock LLM Reranker initialization failed: {e}")
            self.llm = None
    
    def rerank(self, 
               query: str, 
               documents: List[Dict],
               original_scores: Optional[List[float]] = None,
               top_k: Optional[int] = None) -> List[RankedDocument]:
        """
        Rerank documents using Mock LLM evaluation
        
        Args:
            query: User query
            documents: List of documents
            original_scores: Original scores
            top_k: Return top-k documents
            
        Returns:
            List of RankedDocument with LLM-evaluated relevance scores
        """
        if not self.llm or not documents:
            return self._fallback_rerank(query, documents, original_scores, top_k)
        
        try:
            ranked_docs = []
            
            for i, doc in enumerate(documents):
                doc_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
                source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
                original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
                
                # Use LLM to evaluate relevance: ask it to rate relevance from 0-10
                eval_prompt = f"""
Câu hỏi: {query}

Tài liệu: {doc_text[:300]}

Đánh giá độ liên quan của tài liệu này với câu hỏi trên (0-10, 10 = rất liên quan):
Chỉ trả lời với một số từ 0-10, không có giải thích khác.
"""
                
                evaluation = self.llm.generate(eval_prompt, "")
                
                # Extract score from response
                score_text = evaluation.answer.strip()
                try:
                    # Try to extract number from response
                    import re
                    numbers = re.findall(r'\d+', score_text)
                    if numbers:
                        relevance_score = int(numbers[0]) / 10.0  # Convert to 0-1 scale
                    else:
                        # Default score based on query-document similarity
                        relevance_score = self._calculate_text_similarity(query, doc_text)
                except:
                    relevance_score = self._calculate_text_similarity(query, doc_text)
                
                # Ensure score is in 0-1 range
                rerank_score = max(0.0, min(1.0, relevance_score))
                
                ranked_docs.append(RankedDocument(
                    content=doc_text,
                    source=source,
                    original_score=original_score,
                    rerank_score=float(rerank_score),
                    final_score=float(rerank_score) * 0.9 + original_score * 0.1,  # Heavy weight to LLM evaluation
                    reasoning=f"LLM evaluation: {relevance_score:.1%} relevance"
                ))
            
            # Sort by final score
            ranked_docs.sort(key=lambda x: x.final_score, reverse=True)
            
            if top_k:
                ranked_docs = ranked_docs[:top_k]
            
            logger.info(f"✓ Mock LLM reranked {len(documents)} documents - created clear score differences")
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Mock LLM reranking error: {e}")
            return self._fallback_rerank(query, documents, original_scores, top_k)
    
    def _calculate_text_similarity(self, text1: str, text2: str) -> float:
        """Calculate simple text similarity as fallback"""
        from utils.text_processor import clean_text
        
        text1_clean = clean_text(text1).lower()
        text2_clean = clean_text(text2).lower()
        
        words1 = set(text1_clean.split())
        words2 = set(text2_clean.split())
        
        if not words1 or not words2:
            return 0.5
        
        # Jaccard similarity
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.5
    
    def _fallback_rerank(self, query, documents, original_scores=None, top_k=None):
        """Fallback reranking"""
        ranked_docs = []
        for i, doc in enumerate(documents):
            doc_text = doc.get("content", str(doc)) if isinstance(doc, dict) else str(doc)
            source = doc.get("source", "Unknown") if isinstance(doc, dict) else "Unknown"
            original_score = original_scores[i] if original_scores and i < len(original_scores) else 0.0
            
            ranked_docs.append(RankedDocument(
                content=doc_text,
                source=source,
                original_score=original_score,
                rerank_score=original_score,
                final_score=original_score,
                reasoning="Fallback (Mock LLM unavailable)"
            ))
        
        if top_k:
            ranked_docs = ranked_docs[:top_k]
        return ranked_docs


def create_reranker(strategy: str = "hybrid", **kwargs):
    """
    Factory function to create reranker
    
    Args:
        strategy: "cross_encoder", "llm", "mock_llm", "semantic", or "hybrid"
        **kwargs: Additional arguments for specific rerankers
        
    Returns:
        Reranker instance
    """
    if strategy == "cross_encoder":
        return CrossEncoderReranker(**kwargs)
    elif strategy == "llm":
        return LLMReranker(**kwargs)
    elif strategy == "mock_llm":
        return MockLLMReranker(**kwargs)
    elif strategy == "semantic":
        return SemanticCoherenceReranker(**kwargs)
    elif strategy == "hybrid":
        return HybridReranker(**kwargs)
    else:
        logger.warning(f"Unknown strategy {strategy}, using mock_llm instead")
        return MockLLMReranker(**kwargs)

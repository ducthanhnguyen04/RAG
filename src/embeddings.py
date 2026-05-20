"""
Embeddings and Vector Database management
"""
from typing import List, Tuple, Optional
import numpy as np
from pathlib import Path
import pickle
import os
from sentence_transformers import SentenceTransformer
from utils.logger import setup_logger
from config.settings import EMBEDDING_MODEL, VECTOR_DB_PATH

# Disable CUDA to avoid GPU memory issues
os.environ['CUDA_VISIBLE_DEVICES'] = ''

logger = setup_logger(__name__)


class EmbeddingManager:
    """
    Manage document embeddings using sentence-transformers
    """
    
    def __init__(self, model_name: str = EMBEDDING_MODEL):
        """
        Initialize embedding manager
        
        Args:
            model_name: Model name from sentence-transformers
        """
        self.model_name = model_name
        logger.info(f"Loading embedding model: {model_name}")
        # Force CPU mode to avoid CUDA errors
        self.model = SentenceTransformer(model_name, device='cpu')
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"✓ Embedding model loaded on CPU")
        
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for text
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        embedding = self.model.encode(text, convert_to_numpy=True)
        return embedding
    
    def embed_batch(self, texts: List[str], batch_size: int = 32) -> List[np.ndarray]:
        """
        Generate embeddings for multiple texts
        
        Args:
            texts: List of texts
            batch_size: Batch size for processing
            
        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts, 
            batch_size=batch_size,
            convert_to_numpy=True,
            show_progress_bar=True
        )
        return [emb for emb in embeddings]


class SimpleVectorDB:
    """
    Simple in-memory vector database
    For production, use FAISS, Milvus, or Pinecone
    """
    
    def __init__(self, embedding_manager: EmbeddingManager):
        """
        Initialize vector database
        
        Args:
            embedding_manager: EmbeddingManager instance
        """
        self.embedding_manager = embedding_manager
        self.documents: List[str] = []
        self.embeddings: List[np.ndarray] = []
        self.metadata: List[dict] = []
        
    def add_document(self, text: str, metadata: dict = None):
        """
        Add single document
        
        Args:
            text: Document text
            metadata: Document metadata
        """
        embedding = self.embedding_manager.embed_text(text)
        self.documents.append(text)
        self.embeddings.append(embedding)
        self.metadata.append(metadata or {})
        logger.info(f"Added document, total: {len(self.documents)}")
    
    def add_documents(self, texts: List[str], metadatas: List[dict] = None):
        """
        Add multiple documents
        
        Args:
            texts: List of document texts
            metadatas: List of metadata dicts
        """
        embeddings = self.embedding_manager.embed_batch(texts)
        self.documents.extend(texts)
        self.embeddings.extend(embeddings)
        
        if metadatas:
            self.metadata.extend(metadatas)
        else:
            self.metadata.extend([{} for _ in texts])
        
        logger.info(f"Added {len(texts)} documents, total: {len(self.documents)}")
    
    def search(self, query: str, top_k: int = 5) -> List[Tuple[str, float, dict]]:
        """
        Search similar documents
        
        Args:
            query: Query text
            top_k: Number of results
            
        Returns:
            List of (document, score, metadata) tuples
        """
        if not self.documents:
            logger.warning("Vector DB is empty")
            return []
        
        query_embedding = self.embedding_manager.embed_text(query)
        
        # Calculate cosine similarity
        scores = []
        for doc_embedding in self.embeddings:
            # Cosine similarity
            similarity = np.dot(query_embedding, doc_embedding) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(doc_embedding) + 1e-8
            )
            scores.append(similarity)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        results = [
            (self.documents[i], float(scores[i]), self.metadata[i])
            for i in top_indices if scores[i] > 0
        ]
        
        logger.info(f"Search returned {len(results)} results")
        return results
    
    def save(self, path: Path = VECTOR_DB_PATH):
        """
        Save database to disk
        
        Args:
            path: Save path
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        data = {
            'documents': self.documents,
            'embeddings': self.embeddings,
            'metadata': self.metadata
        }
        
        with open(path / 'vector_db.pkl', 'wb') as f:
            pickle.dump(data, f)
        
        logger.info(f"Vector DB saved to {path}")
    
    def load(self, path: Path = VECTOR_DB_PATH):
        """
        Load database from disk
        
        Args:
            path: Load path
        """
        path = Path(path)
        db_file = path / 'vector_db.pkl'
        
        if db_file.exists():
            with open(db_file, 'rb') as f:
                data = pickle.load(f)
            
            self.documents = data['documents']
            self.embeddings = data['embeddings']
            self.metadata = data['metadata']
            logger.info(f"Vector DB loaded from {path}")
        else:
            logger.warning(f"No vector DB found at {path}")
    
    def clear(self):
        """Clear all data"""
        self.documents.clear()
        self.embeddings.clear()
        self.metadata.clear()
        logger.info("Vector DB cleared")

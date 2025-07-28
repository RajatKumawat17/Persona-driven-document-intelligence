"""Embedding engine for generating semantic vectors."""

import os
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Union, Optional
import logging
from pathlib import Path
import torch

from config.settings import MODEL_NAME, MODELS_DIR, BATCH_SIZE

logger = logging.getLogger(__name__)

class EmbeddingEngine:
    """Handles semantic embedding generation and similarity computation."""
    
    def __init__(self):
        self.model: Optional[SentenceTransformer] = None
        self.model_loaded = False
    
    def load_model(self) -> bool:
        """
        Load the sentence transformer model.
        
        Returns:
            Boolean indicating success
        """
        try:
            # Set device to CPU (as per challenge requirements)
            device = 'cpu'
            
            # Check if model exists locally
            local_model_path = MODELS_DIR / MODEL_NAME
            
            if local_model_path.exists():
                logger.info(f"Loading model from local cache: {local_model_path}")
                self.model = SentenceTransformer(str(local_model_path), device=device)
            else:
                logger.info(f"Downloading and caching model: {MODEL_NAME}")
                self.model = SentenceTransformer(MODEL_NAME, device=device)
                # Save model locally for offline use
                self.model.save(str(local_model_path))
                logger.info(f"Model cached at: {local_model_path}")
            
            # Optimize for CPU inference
            if hasattr(self.model, '_modules'):
                for module in self.model._modules.values():
                    if hasattr(module, 'eval'):
                        module.eval()
            
            self.model_loaded = True
            logger.info("Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def generate_embeddings(self, texts: Union[str, List[str]], 
                          show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for input texts.
        
        Args:
            texts: Single text string or list of texts
            show_progress: Whether to show progress bar
            
        Returns:
            Numpy array of embeddings
        """
        if not self.model_loaded:
            if not self.load_model():
                raise RuntimeError("Failed to load embedding model")
        
        try:
            # Ensure texts is a list
            if isinstance(texts, str):
                texts = [texts]
            
            # Filter out empty texts
            texts = [text.strip() for text in texts if text.strip()]
            
            if not texts:
                return np.array([])
            
            # Generate embeddings in batches for memory efficiency
            embeddings = self.model.encode(
                texts,
                batch_size=BATCH_SIZE,
                show_progress_bar=show_progress,
                normalize_embeddings=True,  # L2 normalization for better cosine similarity
                convert_to_numpy=True
            )
            
            logger.debug(f"Generated embeddings for {len(texts)} texts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            raise
    
    def compute_similarity(self, query_embedding: np.ndarray, 
                          doc_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Query embedding vector
            doc_embeddings: Document embedding matrix
            
        Returns:
            Array of similarity scores
        """
        try:
            # Ensure proper dimensions
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)
            
            if doc_embeddings.ndim == 1:
                doc_embeddings = doc_embeddings.reshape(1, -1)
            
            # Compute cosine similarity using normalized vectors
            similarities = np.dot(doc_embeddings, query_embedding.T).flatten()
            
            logger.debug(f"Computed similarities for {len(similarities)} documents")
            return similarities
            
        except Exception as e:
            logger.error(f"Error computing similarity: {str(e)}")
            raise
    
    def create_query_embedding(self, persona: str, job_to_be_done: str) -> np.ndarray:
        """
        Create a rich query embedding from persona and job description.
        
        Args:
            persona: Persona description
            job_to_be_done: Job to be done description
            
        Returns:
            Query embedding vector
        """
        try:
            # Create a rich, contextual query
            query_parts = [
                f"Persona: {persona}",
                f"Task: {job_to_be_done}",
                f"Context: {persona} needs to {job_to_be_done}"
            ]
            
            # Join parts with special tokens for better understanding
            full_query = " [SEP] ".join(query_parts)
            
            # Generate embedding
            query_embedding = self.generate_embeddings(full_query, show_progress=False)
            
            logger.info("Created query embedding from persona and job description")
            return query_embedding
            
        except Exception as e:
            logger.error(f"Error creating query embedding: {str(e)}")
            raise
    
    def batch_similarity_search(self, query_embedding: np.ndarray, 
                               texts: List[str], 
                               top_k: Optional[int] = None) -> List[tuple]:
        """
        Perform batch similarity search.
        
        Args:
            query_embedding: Query embedding vector
            texts: List of texts to search through
            top_k: Number of top results to return (None for all)
            
        Returns:
            List of (index, similarity_score, text) tuples, sorted by relevance
        """
        try:
            if not texts:
                return []
            
            # Generate embeddings for all texts
            doc_embeddings = self.generate_embeddings(texts, show_progress=True)
            
            # Compute similarities
            similarities = self.compute_similarity(query_embedding, doc_embeddings)
            
            # Create results with indices
            results = [(i, sim, texts[i]) for i, sim in enumerate(similarities)]
            
            # Sort by similarity (descending)
            results.sort(key=lambda x: x[1], reverse=True)
            
            # Return top_k results if specified
            if top_k:
                results = results[:top_k]
            
            logger.info(f"Performed batch similarity search on {len(texts)} texts")
            return results
            
        except Exception as e:
            logger.error(f"Error in batch similarity search: {str(e)}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the loaded model."""
        if not self.model_loaded:
            return {"status": "not_loaded"}
        
        return {
            "status": "loaded",
            "model_name": MODEL_NAME,
            "max_seq_length": getattr(self.model, 'max_seq_length', 'unknown'),
            "embedding_dimension": self.model.get_sentence_embedding_dimension(),
            "device": str(self.model.device)
        }
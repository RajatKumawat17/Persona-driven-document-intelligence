"""Extractive summarization module for generating refined text."""

import re
import numpy as np
from typing import List, Tuple, Dict
from dataclasses import dataclass
import logging

from src.embedding_engine import EmbeddingEngine
from config.settings import SUBSECTION_SENTENCES_COUNT

logger = logging.getLogger(__name__)

@dataclass
class SentenceScore:
    """Represents a sentence with its relevance score."""
    text: str
    score: float
    position: int
    length: int

class ExtractiveSummarizer:
    """Generates extractive summaries using semantic similarity."""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
    
    def summarize_section(self, content: str, query_embedding: np.ndarray, 
                         max_sentences: int = None) -> str:
        """
        Generate an extractive summary of a section.
        
        Args:
            content: Section content to summarize
            query_embedding: Query embedding vector
            max_sentences: Maximum number of sentences to extract
            
        Returns:
            Summarized text
        """
        try:
            if not content.strip():
                return ""
            
            max_sentences = max_sentences or SUBSECTION_SENTENCES_COUNT
            
            # Split content into sentences
            sentences = self._split_into_sentences(content)
            
            if len(sentences) <= max_sentences:
                return content.strip()
            
            # Score sentences
            sentence_scores = self._score_sentences(sentences, query_embedding)
            
            # Select top sentences
            selected_sentences = self._select_top_sentences(
                sentence_scores, max_sentences
            )
            
            # Reconstruct summary maintaining original order
            summary = self._reconstruct_summary(selected_sentences)
            
            logger.debug(f"Summarized {len(sentences)} sentences to {len(selected_sentences)}")
            return summary
            
        except Exception as e:
            logger.error(f"Error summarizing section: {str(e)}")
            return content[:500] + "..." if len(content) > 500 else content
    
    def _split_into_sentences(self, content: str) -> List[str]:
        """Split content into sentences using regex patterns."""
        # Clean content
        content = re.sub(r'\s+', ' ', content.strip())
        
        # Split into sentences using multiple delimiters
        sentence_endings = r'[.!?]+(?:\s+|$)'
        sentences = re.split(sentence_endings, content)
        
        # Clean and filter sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Filter out very short sentences or single words
            if len(sentence) > 10 and len(sentence.split()) > 3:
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def _score_sentences(self, sentences: List[str], 
                        query_embedding: np.ndarray) -> List[SentenceScore]:
        """Score sentences based on relevance to query."""
        if not sentences:
            return []
        
        try:
            # Generate embeddings for all sentences
            sentence_embeddings = self.embedding_engine.generate_embeddings(
                sentences, show_progress=False
            )
            
            # Compute similarities
            similarities = self.embedding_engine.compute_similarity(
                query_embedding, sentence_embeddings
            )
            
            # Create sentence scores with additional features
            sentence_scores = []
            for i, (sentence, similarity) in enumerate(zip(sentences, similarities)):
                # Additional scoring factors
                position_score = self._calculate_position_score(i, len(sentences))
                length_score = self._calculate_length_score(sentence)
                keyword_score = self._calculate_keyword_score(sentence)
                
                # Combined score with weights
                combined_score = (
                    0.7 * similarity +           # Semantic similarity (primary)
                    0.15 * position_score +      # Position importance
                    0.1 * length_score +         # Length preference
                    0.05 * keyword_score         # Keyword presence
                )
                
                sentence_score = SentenceScore(
                    text=sentence,
                    score=float(combined_score),
                    position=i,
                    length=len(sentence)
                )
                sentence_scores.append(sentence_score)
            
            return sentence_scores
            
        except Exception as e:
            logger.error(f"Error scoring sentences: {str(e)}")
            # Fallback: return sentences with uniform scores
            return [
                SentenceScore(text=sent, score=0.5, position=i, length=len(sent))
                for i, sent in enumerate(sentences)
            ]
    
    def _calculate_position_score(self, position: int, total_sentences: int) -> float:
        """Calculate position-based score (beginning and end are more important)."""
        if total_sentences <= 1:
            return 1.0
        
        # Normalize position to 0-1
        normalized_pos = position / (total_sentences - 1)
        
        # U-shaped curve: higher scores for beginning and end
        position_score = 1 - (2 * abs(normalized_pos - 0.5))
        return max(0.1, position_score)  # Minimum score of 0.1
    
    def _calculate_length_score(self, sentence: str) -> float:
        """Calculate length-based score (prefer medium-length sentences)."""
        length = len(sentence)
        
        # Optimal length range (50-150 characters)
        if 50 <= length <= 150:
            return 1.0
        elif 30 <= length <= 200:
            return 0.8
        elif 20 <= length <= 250:
            return 0.6
        else:
            return 0.3
    
    def _calculate_keyword_score(self, sentence: str) -> float:
        """Calculate keyword-based score (presence of important terms)."""
        # Common important terms that might indicate key information
        important_terms = [
            'result', 'conclusion', 'finding', 'analysis', 'study', 'research',
            'important', 'significant', 'key', 'main', 'primary', 'critical',
            'method', 'approach', 'technique', 'model', 'framework',
            'performance', 'accuracy', 'effectiveness', 'efficiency'
        ]
        
        sentence_lower = sentence.lower()
        keyword_count = sum(1 for term in important_terms if term in sentence_lower)
        
        # Normalize score based on keyword density
        keyword_density = keyword_count / len(sentence.split()) if sentence.split() else 0
        return min(1.0, keyword_density * 10)  # Cap at 1.0
    
    def _select_top_sentences(self, sentence_scores: List[SentenceScore], 
                             max_sentences: int) -> List[SentenceScore]:
        """Select top sentences avoiding redundancy."""
        if len(sentence_scores) <= max_sentences:
            return sentence_scores
        
        # Sort by score (descending)
        sorted_sentences = sorted(sentence_scores, key=lambda x: x.score, reverse=True)
        
        # Select top sentences with diversity consideration
        selected = []
        used_positions = set()
        
        for sentence_score in sorted_sentences:
            if len(selected) >= max_sentences:
                break
            
            # Avoid selecting adjacent sentences to improve diversity
            if not any(abs(sentence_score.position - pos) < 2 for pos in used_positions):
                selected.append(sentence_score)
                used_positions.add(sentence_score.position)
        
        # If we don't have enough sentences due to diversity constraint,
        # add remaining top sentences
        if len(selected) < max_sentences:
            for sentence_score in sorted_sentences:
                if sentence_score not in selected and len(selected) < max_sentences:
                    selected.append(sentence_score)
        
        return selected
    
    def _reconstruct_summary(self, selected_sentences: List[SentenceScore]) -> str:
        """Reconstruct summary maintaining original sentence order."""
        # Sort by original position to maintain flow
        sorted_by_position = sorted(selected_sentences, key=lambda x: x.position)
        
        # Join sentences
        summary_text = '. '.join(sentence.text for sentence in sorted_by_position)
        
        # Clean up the summary
        summary_text = re.sub(r'\s+', ' ', summary_text.strip())
        
        # Ensure proper ending
        if not summary_text.endswith(('.', '!', '?')):
            summary_text += '.'
        
        return summary_text
    
    def batch_summarize(self, contents: List[str], query_embedding: np.ndarray,
                       max_sentences: int = None) -> List[str]:
        """
        Batch summarize multiple content pieces.
        
        Args:
            contents: List of content strings to summarize
            query_embedding: Query embedding vector
            max_sentences: Maximum sentences per summary
            
        Returns:
            List of summarized texts
        """
        try:
            summaries = []
            for content in contents:
                summary = self.summarize_section(content, query_embedding, max_sentences)
                summaries.append(summary)
            
            logger.info(f"Batch summarized {len(contents)} content pieces")
            return summaries
            
        except Exception as e:
            logger.error(f"Error in batch summarization: {str(e)}")
            return contents  # Return original contents as fallback
    
    def get_summary_statistics(self, original_content: str, summary: str) -> Dict:
        """Get statistics about the summarization."""
        try:
            original_sentences = len(self._split_into_sentences(original_content))
            summary_sentences = len(self._split_into_sentences(summary))
            
            return {
                "original_length": len(original_content),
                "summary_length": len(summary),
                "compression_ratio": len(summary) / len(original_content) if original_content else 0,
                "original_sentences": original_sentences,
                "summary_sentences": summary_sentences,
                "sentence_reduction": (original_sentences - summary_sentences) / original_sentences if original_sentences else 0
            }
            
        except Exception as e:
            logger.error(f"Error calculating summary statistics: {str(e)}")
            return {}
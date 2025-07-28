"""Relevance ranking module for analyzing document sections."""

import numpy as np
from typing import List, Dict, Tuple
from dataclasses import dataclass
import logging

from src.pdf_processor import DocumentSection
from src.embedding_engine import EmbeddingEngine
from config.settings import (
    TOP_SECTIONS_COUNT,
    SIMILARITY_THRESHOLD
)

logger = logging.getLogger(__name__)

@dataclass
class RankedSection:
    """Represents a ranked document section with relevance score."""
    section: DocumentSection
    relevance_score: float
    importance_rank: int

class RelevanceRanker:
    """Ranks document sections based on relevance to persona and job."""
    
    def __init__(self, embedding_engine: EmbeddingEngine):
        self.embedding_engine = embedding_engine
    
    def rank_sections(self, sections: List[DocumentSection], 
                     persona: str, job_to_be_done: str) -> List[RankedSection]:
        """
        Rank sections based on relevance to persona and job.
        
        Args:
            sections: List of document sections
            persona: Persona description
            job_to_be_done: Job to be done description
            
        Returns:
            List of ranked sections sorted by relevance
        """
        try:
            if not sections:
                logger.warning("No sections provided for ranking")
                return []
            
            # Create query embedding
            query_embedding = self.embedding_engine.create_query_embedding(
                persona, job_to_be_done
            )
            
            # Prepare section texts for embedding
            section_texts = []
            for section in sections:
                # Combine title and content for better context
                combined_text = f"{section.title}: {section.content[:1000]}"  # Limit content length
                section_texts.append(combined_text)
            
            # Generate embeddings for all sections
            logger.info(f"Generating embeddings for {len(sections)} sections")
            section_embeddings = self.embedding_engine.generate_embeddings(
                section_texts, show_progress=True
            )
            
            # Compute similarities
            similarities = self.embedding_engine.compute_similarity(
                query_embedding, section_embeddings
            )
            
            # Create ranked sections
            ranked_sections = []
            for i, (section, similarity) in enumerate(zip(sections, similarities)):
                if similarity >= SIMILARITY_THRESHOLD:  # Filter out irrelevant sections
                    ranked_section = RankedSection(
                        section=section,
                        relevance_score=float(similarity),
                        importance_rank=0  # Will be set after sorting
                    )
                    ranked_sections.append(ranked_section)
            
            # Sort by relevance score (descending)
            ranked_sections.sort(key=lambda x: x.relevance_score, reverse=True)
            
            # Assign importance ranks
            for rank, ranked_section in enumerate(ranked_sections, 1):
                ranked_section.importance_rank = rank
            
            # Limit to top sections
            ranked_sections = ranked_sections[:TOP_SECTIONS_COUNT]
            
            logger.info(f"Ranked {len(ranked_sections)} relevant sections")
            return ranked_sections
            
        except Exception as e:
            logger.error(f"Error ranking sections: {str(e)}")
            raise
    
    def analyze_section_diversity(self, ranked_sections: List[RankedSection]) -> Dict:
        """
        Analyze diversity of ranked sections across documents and levels.
        
        Args:
            ranked_sections: List of ranked sections
            
        Returns:
            Dictionary with diversity statistics
        """
        try:
            if not ranked_sections:
                return {}
            
            # Document distribution
            doc_counts = {}
            level_counts = {}
            page_distribution = []
            
            for ranked_section in ranked_sections:
                section = ranked_section.section
                
                # Count by document
                doc_name = section.document_name
                doc_counts[doc_name] = doc_counts.get(doc_name, 0) + 1
                
                # Count by heading level
                level = section.level
                level_counts[level] = level_counts.get(level, 0) + 1
                
                # Track page numbers
                page_distribution.append(section.page_number)
            
            # Calculate statistics
            total_sections = len(ranked_sections)
            unique_documents = len(doc_counts)
            
            diversity_stats = {
                "total_sections": total_sections,
                "unique_documents": unique_documents,
                "document_distribution": doc_counts,
                "level_distribution": level_counts,
                "page_range": {
                    "min": min(page_distribution),
                    "max": max(page_distribution),
                    "avg": sum(page_distribution) / len(page_distribution)
                },
                "avg_relevance_score": sum(rs.relevance_score for rs in ranked_sections) / total_sections,
                "relevance_score_range": {
                    "min": min(rs.relevance_score for rs in ranked_sections),
                    "max": max(rs.relevance_score for rs in ranked_sections)
                }
            }
            
            logger.info("Computed section diversity statistics")
            return diversity_stats
            
        except Exception as e:
            logger.error(f"Error analyzing section diversity: {str(e)}")
            return {}
    
    def filter_sections_by_criteria(self, ranked_sections: List[RankedSection],
                                   min_score: float = None,
                                   required_documents: List[str] = None,
                                   required_levels: List[str] = None) -> List[RankedSection]:
        """
        Filter ranked sections based on specific criteria.
        
        Args:
            ranked_sections: List of ranked sections
            min_score: Minimum relevance score threshold
            required_documents: List of required document names
            required_levels: List of required heading levels
            
        Returns:
            Filtered list of ranked sections
        """
        try:
            filtered_sections = ranked_sections.copy()
            
            # Filter by minimum score
            if min_score is not None:
                filtered_sections = [
                    rs for rs in filtered_sections 
                    if rs.relevance_score >= min_score
                ]
            
            # Filter by required documents
            if required_documents:
                filtered_sections = [
                    rs for rs in filtered_sections 
                    if rs.section.document_name in required_documents
                ]
            
            # Filter by required levels
            if required_levels:
                filtered_sections = [
                    rs for rs in filtered_sections 
                    if rs.section.level in required_levels
                ]
            
            # Re-assign importance ranks
            for rank, ranked_section in enumerate(filtered_sections, 1):
                ranked_section.importance_rank = rank
            
            logger.info(f"Filtered sections: {len(ranked_sections)} -> {len(filtered_sections)}")
            return filtered_sections
            
        except Exception as e:
            logger.error(f"Error filtering sections: {str(e)}")
            return ranked_sections
    
    def get_ranking_insights(self, ranked_sections: List[RankedSection], 
                           persona: str, job_to_be_done: str) -> Dict:
        """
        Generate insights about the ranking results.
        
        Args:
            ranked_sections: List of ranked sections
            persona: Persona description
            job_to_be_done: Job description
            
        Returns:
            Dictionary with ranking insights
        """
        try:
            if not ranked_sections:
                return {"message": "No sections to analyze"}
            
            # Top section analysis
            top_section = ranked_sections[0]
            
            # Score distribution
            scores = [rs.relevance_score for rs in ranked_sections]
            
            insights = {
                "query_context": {
                    "persona": persona,
                    "job_to_be_done": job_to_be_done
                },
                "top_match": {
                    "title": top_section.section.title,
                    "document": top_section.section.document_name,
                    "score": top_section.relevance_score,
                    "page": top_section.section.page_number
                },
                "score_statistics": {
                    "mean": sum(scores) / len(scores),
                    "median": sorted(scores)[len(scores) // 2],
                    "std": np.std(scores),
                    "range": max(scores) - min(scores)
                },
                "coverage_analysis": self.analyze_section_diversity(ranked_sections),
                "recommendations": self._generate_recommendations(ranked_sections)
            }
            
            return insights
            
        except Exception as e:
            logger.error(f"Error generating ranking insights: {str(e)}")
            return {"error": str(e)}
    
    def _generate_recommendations(self, ranked_sections: List[RankedSection]) -> List[str]:
        """Generate recommendations based on ranking results."""
        recommendations = []
        
        if not ranked_sections:
            return ["No relevant sections found. Consider refining the persona or job description."]
        
        # Check score distribution
        scores = [rs.relevance_score for rs in ranked_sections]
        avg_score = sum(scores) / len(scores)
        
        if avg_score < 0.3:
            recommendations.append("Low relevance scores detected. Consider refining the query or adding more specific keywords.")
        
        # Check document diversity
        unique_docs = len(set(rs.section.document_name for rs in ranked_sections))
        if unique_docs == 1:
            recommendations.append("Results are concentrated in one document. Consider checking if other documents contain relevant information.")
        
        # Check level diversity
        levels = [rs.section.level for rs in ranked_sections]
        if all(level == levels[0] for level in levels):
            recommendations.append(f"All results are from {levels[0]} level. Consider exploring other heading levels for broader context.")
        
        return recommendations if recommendations else ["Good coverage and relevance across documents."]
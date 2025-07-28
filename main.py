#!/usr/bin/env python3
"""
Main script for Persona-Driven Document Intelligence.
This is the Docker entry point for the hackathon challenge.
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from src.pdf_processor import PDFProcessor
from src.embedding_engine import EmbeddingEngine
from src.relevance_ranker import RelevanceRanker
from src.summarizer import ExtractiveSummarizer
from src.output_formatter import OutputFormatter
from config.settings import INPUT_DIR, OUTPUT_DIR, MAX_PROCESSING_TIME_SECONDS

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIntelligenceSystem:
    """Main system for persona-driven document intelligence."""
    
    def __init__(self):
        self.pdf_processor = PDFProcessor()
        self.embedding_engine = EmbeddingEngine()
        self.relevance_ranker = None
        self.summarizer = None
        self.output_formatter = OutputFormatter()
        self.processing_stats = {}
    
    def initialize_components(self) -> bool:
        """Initialize all system components."""
        try:
            logger.info("Initializing system components...")
            
            # Load embedding model
            start_time = time.time()
            if not self.embedding_engine.load_model():
                logger.error("Failed to load embedding model")
                return False
            self.processing_stats['model_load_time'] = time.time() - start_time
            
            # Initialize dependent components
            self.relevance_ranker = RelevanceRanker(self.embedding_engine)
            self.summarizer = ExtractiveSummarizer(self.embedding_engine)
            
            logger.info("System components initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing components: {str(e)}")
            return False
    
    def load_input_configuration(self) -> Dict[str, Any]:
        """Load input configuration from the input directory."""
        try:
            # Look for configuration file
            config_files = list(INPUT_DIR.glob("*.json"))
            if not config_files:
                logger.error("No configuration file found in input directory")
                return None
            
            config_path = config_files[0]  # Use first JSON file found
            
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            
            logger.info(f"Loaded configuration from: {config_path}")
            return config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            return None
    
    def find_pdf_files(self) -> List[str]:
        """Find all PDF files in the input directory."""
        try:
            pdf_files = list(INPUT_DIR.glob("*.pdf"))
            pdf_paths = [str(pdf_file) for pdf_file in pdf_files]
            
            logger.info(f"Found {len(pdf_paths)} PDF files")
            return pdf_paths
            
        except Exception as e:
            logger.error(f"Error finding PDF files: {str(e)}")
            return []
    
    def process_documents(self, pdf_paths: List[str]) -> List[Any]:
        """Process all PDF documents and extract sections."""
        try:
            start_time = time.time()
            all_sections = []
            
            for pdf_path in pdf_paths:
                logger.info(f"Processing: {pdf_path}")
                sections = self.pdf_processor.process_pdf(pdf_path)
                all_sections.extend(sections)
            
            self.processing_stats['pdf_processing_time'] = time.time() - start_time
            logger.info(f"Processed {len(pdf_paths)} documents, extracted {len(all_sections)} sections")
            
            return all_sections
            
        except Exception as e:
            logger.error(f"Error processing documents: {str(e)}")
            return []
    
    def analyze_relevance(self, sections: List[Any], persona: str, job_to_be_done: str) -> List[Any]:
        """Analyze section relevance based on persona and job."""
        try:
            start_time = time.time()
            
            ranked_sections = self.relevance_ranker.rank_sections(
                sections, persona, job_to_be_done
            )
            
            self.processing_stats['ranking_time'] = time.time() - start_time
            logger.info(f"Ranked {len(ranked_sections)} relevant sections")
            
            return ranked_sections
            
        except Exception as e:
            logger.error(f"Error analyzing relevance: {str(e)}")
            return []
    
    def generate_summaries(self, ranked_sections: List[Any], 
                          persona: str, job_to_be_done: str) -> Dict[int, str]:
        """Generate extractive summaries for top sections."""
        try:
            start_time = time.time()
            
            # Create query embedding for summarization
            query_embedding = self.embedding_engine.create_query_embedding(
                persona, job_to_be_done
            )
            
            summaries = {}
            for i, ranked_section in enumerate(ranked_sections):
                summary = self.summarizer.summarize_section(
                    ranked_section.section.content, query_embedding
                )
                summaries[i] = summary
            
            self.processing_stats['summarization_time'] = time.time() - start_time
            logger.info(f"Generated summaries for {len(summaries)} sections")
            
            return summaries
            
        except Exception as e:
            logger.error(f"Error generating summaries: {str(e)}")
            return {}
    
    def run_analysis(self) -> bool:
        """Run the complete analysis pipeline."""
        try:
            total_start_time = time.time()
            
            # Initialize components
            if not self.initialize_components():
                return False
            
            # Load configuration
            config = self.load_input_configuration()
            if not config:
                logger.error("Failed to load configuration")
                return False
            
            # Extract configuration
            persona = config.get('persona', '')
            job_to_be_done = config.get('job_to_be_done', '')
            
            if not persona or not job_to_be_done:
                logger.error("Missing persona or job_to_be_done in configuration")
                return False
            
            logger.info(f"Persona: {persona}")
            logger.info(f"Job to be done: {job_to_be_done}")
            
            # Find PDF files
            pdf_paths = self.find_pdf_files()
            if not pdf_paths:
                logger.error("No PDF files found")
                return False
            
            # Process documents
            sections = self.process_documents(pdf_paths)
            if not sections:
                logger.error("No sections extracted from documents")
                return False
            
            # Analyze relevance
            ranked_sections = self.analyze_relevance(sections, persona, job_to_be_done)
            if not ranked_sections:
                logger.error("No relevant sections found")
                return False
            
            # Generate summaries
            summaries = self.generate_summaries(ranked_sections, persona, job_to_be_done)
            
            # Format and save results
            input_documents = [Path(path).name for path in pdf_paths]
            
            # Add total processing time
            self.processing_stats['total_time'] = time.time() - total_start_time
            
            results = self.output_formatter.format_results(
                ranked_sections=ranked_sections,
                summarized_sections=summaries,
                input_documents=input_documents,
                persona=persona,
                job_to_be_done=job_to_be_done,
                processing_stats=self.processing_stats
            )
            
            # Validate output format
            validation_errors = self.output_formatter.validate_output_format(results)
            if validation_errors:
                logger.warning(f"Output validation warnings: {validation_errors}")
            
            # Save results
            output_path = self.output_formatter.save_results(results)
            logger.info(f"Analysis completed successfully. Results saved to: {output_path}")
            
            # Print summary
            total_time = self.processing_stats['total_time']
            logger.info(f"Total processing time: {total_time:.2f} seconds")
            logger.info(f"Sections analyzed: {len(sections)}")
            logger.info(f"Relevant sections found: {len(ranked_sections)}")
            logger.info(f"Summaries generated: {len(summaries)}")
            
            # Check time constraint
            if total_time > MAX_PROCESSING_TIME_SECONDS:
                logger.warning(f"Processing time ({total_time:.2f}s) exceeded limit ({MAX_PROCESSING_TIME_SECONDS}s)")
            
            return True
            
        except Exception as e:
            logger.error(f"Error during analysis: {str(e)}")
            return False

def main():
    """Main entry point for the application."""
    try:
        logger.info("Starting Persona-Driven Document Intelligence System")
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Initialize and run system
        system = DocumentIntelligenceSystem()
        success = system.run_analysis()
        
        if success:
            logger.info("Analysis completed successfully")
            sys.exit(0)
        else:
            logger.error("Analysis failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()
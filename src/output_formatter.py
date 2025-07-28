"""Output formatting module for generating JSON results."""

import json
from datetime import datetime
from typing import List, Dict, Any
from pathlib import Path
import logging

from src.relevance_ranker import RankedSection
from config.settings import TIMESTAMP_FORMAT, OUTPUT_DIR, OUTPUT_FILENAME

logger = logging.getLogger(__name__)

class OutputFormatter:
    """Formats analysis results into the required JSON output format."""
    
    def __init__(self):
        pass
    
    def format_results(self, 
                      ranked_sections: List[RankedSection],
                      summarized_sections: Dict[int, str],
                      input_documents: List[str],
                      persona: str,
                      job_to_be_done: str,
                      processing_stats: Dict = None) -> Dict[str, Any]:
        """
        Format results into the required JSON structure.
        
        Args:
            ranked_sections: List of ranked document sections
            summarized_sections: Dictionary mapping section indices to summaries
            input_documents: List of input document names
            persona: Persona description
            job_to_be_done: Job to be done description
            processing_stats: Optional processing statistics
            
        Returns:
            Formatted results dictionary
        """
        try:
            # Generate timestamp
            timestamp = datetime.now().strftime(TIMESTAMP_FORMAT)
            
            # Build metadata
            metadata = {
                "input_documents": input_documents,
                "persona": persona,
                "job_to_be_done": job_to_be_done,
                "processing_timestamp": timestamp,
                "total_sections_analyzed": len(ranked_sections),
                "processing_stats": processing_stats or {}
            }
            
            # Build extracted sections
            extracted_sections = []
            for ranked_section in ranked_sections:
                section_data = {
                    "document": ranked_section.section.document_name,
                    "page_number": ranked_section.section.page_number,
                    "section_title": ranked_section.section.title,
                    "importance_rank": ranked_section.importance_rank,
                    "relevance_score": round(ranked_section.relevance_score, 4),
                    "section_level": ranked_section.section.level
                }
                extracted_sections.append(section_data)
            
            # Build sub-section analysis
            subsection_analysis = []
            for i, ranked_section in enumerate(ranked_sections):
                refined_text = summarized_sections.get(i, "")
                
                if refined_text:  # Only include if we have summarized text
                    subsection_data = {
                        "document": ranked_section.section.document_name,
                        "page_number": ranked_section.section.page_number,
                        "refined_text": refined_text,
                        "importance_rank": ranked_section.importance_rank
                    }
                    subsection_analysis.append(subsection_data)
            
            # Combine all results
            results = {
                "metadata": metadata,
                "extracted_sections": extracted_sections,
                "subsection_analysis": subsection_analysis
            }
            
            logger.info("Successfully formatted results")
            return results
            
        except Exception as e:
            logger.error(f"Error formatting results: {str(e)}")
            raise
    
    def save_results(self, results: Dict[str, Any], 
                    output_path: str = None) -> str:
        """
        Save results to JSON file.
        
        Args:
            results: Formatted results dictionary
            output_path: Optional custom output path
            
        Returns:
            Path to saved file
        """
        try:
            if output_path is None:
                output_path = OUTPUT_DIR / OUTPUT_FILENAME
            else:
                output_path = Path(output_path)
            
            # Ensure output directory exists
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save JSON with proper formatting
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"Results saved to: {output_path}")
            return str(output_path)
            
        except Exception as e:
            logger.error(f"Error saving results: {str(e)}")
            raise
    
    def validate_output_format(self, results: Dict[str, Any]) -> List[str]:
        """
        Validate output format against requirements.
        
        Args:
            results: Results dictionary to validate
            
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        try:
            # Check top-level structure
            required_keys = ["metadata", "extracted_sections", "subsection_analysis"]
            for key in required_keys:
                if key not in results:
                    errors.append(f"Missing required key: {key}")
            
            # Validate metadata
            if "metadata" in results:
                metadata = results["metadata"]
                metadata_required = ["input_documents", "persona", "job_to_be_done", "processing_timestamp"]
                for key in metadata_required:
                    if key not in metadata:
                        errors.append(f"Missing metadata key: {key}")
            
            # Validate extracted sections
            if "extracted_sections" in results:
                sections = results["extracted_sections"]
                if not isinstance(sections, list):
                    errors.append("extracted_sections must be a list")
                else:
                    section_required = ["document", "page_number", "section_title", "importance_rank"]
                    for i, section in enumerate(sections):
                        for key in section_required:
                            if key not in section:
                                errors.append(f"Missing key '{key}' in extracted_sections[{i}]")
            
            # Validate subsection analysis
            if "subsection_analysis" in results:
                subsections = results["subsection_analysis"]
                if not isinstance(subsections, list):
                    errors.append("subsection_analysis must be a list")
                else:
                    subsection_required = ["document", "page_number", "refined_text", "importance_rank"]
                    for i, subsection in enumerate(subsections):
                        for key in subsection_required:
                            if key not in subsection:
                                errors.append(f"Missing key '{key}' in subsection_analysis[{i}]")
            
            logger.info(f"Output validation completed. Errors found: {len(errors)}")
            return errors
            
        except Exception as e:
            logger.error(f"Error during validation: {str(e)}")
            return [f"Validation error: {str(e)}"]
    
    def create_summary_report(self, results: Dict[str, Any]) -> str:
        """
        Create a human-readable summary report.
        
        Args:
            results: Formatted results dictionary
            
        Returns:
            Summary report as string
        """
        try:
            if not results:
                return "No results to summarize."
            
            metadata = results.get("metadata", {})
            extracted_sections = results.get("extracted_sections", [])
            subsection_analysis = results.get("subsection_analysis", [])
            
            report_lines = [
                "=== PERSONA-DRIVEN DOCUMENT INTELLIGENCE REPORT ===",
                "",
                f"Persona: {metadata.get('persona', 'N/A')}",
                f"Job to be Done: {metadata.get('job_to_be_done', 'N/A')}",
                f"Processing Time: {metadata.get('processing_timestamp', 'N/A')}",
                f"Documents Analyzed: {len(metadata.get('input_documents', []))}",
                "",
                "=== INPUT DOCUMENTS ===",
            ]
            
            for doc in metadata.get('input_documents', []):
                report_lines.append(f"- {doc}")
            
            report_lines.extend([
                "",
                f"=== TOP RELEVANT SECTIONS ({len(extracted_sections)}) ===",
                ""
            ])
            
            for section in extracted_sections[:5]:  # Show top 5
                report_lines.extend([
                    f"Rank {section['importance_rank']}: {section['section_title']}",
                    f"  Document: {section['document']} (Page {section['page_number']})",
                    f"  Relevance Score: {section.get('relevance_score', 'N/A')}",
                    f"  Level: {section.get('section_level', 'N/A')}",
                    ""
                ])
            
            if len(extracted_sections) > 5:
                report_lines.append(f"... and {len(extracted_sections) - 5} more sections")
            
            report_lines.extend([
                "",
                f"=== REFINED ANALYSIS ({len(subsection_analysis)} sections) ===",
                ""
            ])
            
            for i, subsection in enumerate(subsection_analysis[:3]):  # Show top 3
                report_lines.extend([
                    f"Section {subsection['importance_rank']} Summary:",
                    f"  Document: {subsection['document']} (Page {subsection['page_number']})",
                    f"  Text: {subsection['refined_text'][:200]}{'...' if len(subsection['refined_text']) > 200 else ''}",
                    ""
                ])
            
            # Statistics
            if "processing_stats" in metadata:
                stats = metadata["processing_stats"]
                report_lines.extend([
                    "=== PROCESSING STATISTICS ===",
                    f"Total Processing Time: {stats.get('total_time', 'N/A')}",
                    f"PDF Processing Time: {stats.get('pdf_processing_time', 'N/A')}",
                    f"Embedding Time: {stats.get('embedding_time', 'N/A')}",
                    f"Ranking Time: {stats.get('ranking_time', 'N/A')}",
                    f"Summarization Time: {stats.get('summarization_time', 'N/A')}",
                    ""
                ])
            
            report_lines.append("=== END REPORT ===")
            
            return "\n".join(report_lines)
            
        except Exception as e:
            logger.error(f"Error creating summary report: {str(e)}")
            return f"Error generating report: {str(e)}"
    
    def export_to_multiple_formats(self, results: Dict[str, Any], 
                                  output_dir: str = None) -> Dict[str, str]:
        """
        Export results to multiple formats.
        
        Args:
            results: Formatted results dictionary
            output_dir: Output directory path
            
        Returns:
            Dictionary with format names and file paths
        """
        try:
            if output_dir is None:
                output_dir = OUTPUT_DIR
            else:
                output_dir = Path(output_dir)
            
            output_dir.mkdir(parents=True, exist_ok=True)
            
            exported_files = {}
            
            # JSON format (main output)
            json_path = output_dir / "analysis_results.json"
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            exported_files['json'] = str(json_path)
            
            # Summary report
            report_path = output_dir / "summary_report.txt"
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(self.create_summary_report(results))
            exported_files['report'] = str(report_path)
            
            # Compact JSON (for API usage)
            compact_path = output_dir / "analysis_results_compact.json"
            with open(compact_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, separators=(',', ':'), ensure_ascii=False)
            exported_files['compact_json'] = str(compact_path)
            
            logger.info(f"Exported results to {len(exported_files)} formats")
            return exported_files
            
        except Exception as e:
            logger.error(f"Error exporting to multiple formats: {str(e)}")
            return {}
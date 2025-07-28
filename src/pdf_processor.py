"""PDF processing module for extracting structure and content."""

import fitz  # PyMuPDF
import re
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import logging

from config.settings import (
    HEADING_FONT_SIZE_THRESHOLD,
    MIN_HEADING_LENGTH,
    MAX_HEADING_LENGTH,
    MIN_SECTION_LENGTH
)

logger = logging.getLogger(__name__)

@dataclass
class DocumentSection:
    """Represents a section of a document."""
    title: str
    content: str
    page_number: int
    level: str  # H1, H2, H3, or content
    font_size: float = 0.0
    document_name: str = ""

class PDFProcessor:
    """Processes PDF files to extract structured content and sections."""
    
    def __init__(self):
        self.font_size_stats = {}
    
    def process_pdf(self, pdf_path: str) -> List[DocumentSection]:
        """
        Process a PDF file and extract structured sections.
        
        Args:
            pdf_path: Path to the PDF file
            
        Returns:
            List of DocumentSection objects
        """
        try:
            doc = fitz.open(pdf_path)
            document_name = pdf_path.split('/')[-1].replace('.pdf', '')
            
            # First pass: analyze font sizes
            self._analyze_font_sizes(doc)
            
            # Second pass: extract structured content
            sections = self._extract_sections(doc, document_name)
            
            doc.close()
            logger.info(f"Extracted {len(sections)} sections from {document_name}")
            return sections
            
        except Exception as e:
            logger.error(f"Error processing PDF {pdf_path}: {str(e)}")
            return []
    
    def _analyze_font_sizes(self, doc: fitz.Document) -> None:
        """Analyze font sizes in the document to identify heading patterns."""
        font_sizes = []
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                    
                for line in block["lines"]:
                    for span in line["spans"]:
                        text = span["text"].strip()
                        if len(text) >= MIN_HEADING_LENGTH:
                            font_sizes.append(span["size"])
        
        if font_sizes:
            font_sizes.sort(reverse=True)
            # Use percentiles to identify heading levels
            self.font_size_stats = {
                'h1_threshold': font_sizes[int(len(font_sizes) * 0.05)] if font_sizes else 12,
                'h2_threshold': font_sizes[int(len(font_sizes) * 0.15)] if font_sizes else 11,
                'h3_threshold': font_sizes[int(len(font_sizes) * 0.30)] if font_sizes else 10,
                'body_avg': sum(font_sizes[int(len(font_sizes) * 0.5):]) / max(1, len(font_sizes[int(len(font_sizes) * 0.5):]))
            }
    
    def _extract_sections(self, doc: fitz.Document, document_name: str) -> List[DocumentSection]:
        """Extract sections from the document based on headings."""
        sections = []
        current_section = None
        
        for page_num in range(len(doc)):
            page = doc[page_num]
            blocks = page.get_text("dict")["blocks"]
            
            for block in blocks:
                if "lines" not in block:
                    continue
                
                for line in block["lines"]:
                    line_text = ""
                    max_font_size = 0
                    
                    # Combine spans in the line
                    for span in line["spans"]:
                        line_text += span["text"]
                        max_font_size = max(max_font_size, span["size"])
                    
                    line_text = line_text.strip()
                    if not line_text or len(line_text) < MIN_HEADING_LENGTH:
                        continue
                    
                    # Determine if this is a heading
                    heading_level = self._classify_heading(line_text, max_font_size)
                    
                    if heading_level:
                        # Save previous section if it exists
                        if current_section and len(current_section.content.strip()) > MIN_SECTION_LENGTH:
                            sections.append(current_section)
                        
                        # Start new section
                        current_section = DocumentSection(
                            title=line_text,
                            content="",
                            page_number=page_num + 1,
                            level=heading_level,
                            font_size=max_font_size,
                            document_name=document_name
                        )
                    else:
                        # Add to current section content
                        if current_section:
                            current_section.content += line_text + "\n"
        
        # Add the last section
        if current_section and len(current_section.content.strip()) > MIN_SECTION_LENGTH:
            sections.append(current_section)
        
        return self._post_process_sections(sections)
    
    def _classify_heading(self, text: str, font_size: float) -> Optional[str]:
        """Classify text as a heading level based on various heuristics."""
        if len(text) > MAX_HEADING_LENGTH:
            return None
        
        # Font size based classification
        if font_size >= self.font_size_stats.get('h1_threshold', 14):
            return 'H1'
        elif font_size >= self.font_size_stats.get('h2_threshold', 12):
            return 'H2'
        elif font_size >= self.font_size_stats.get('h3_threshold', 11):
            return 'H3'
        
        # Pattern-based classification (backup method)
        if self._is_heading_by_pattern(text):
            # Use relative font size if available
            body_avg = self.font_size_stats.get('body_avg', 10)
            if font_size > body_avg * 1.5:
                return 'H1'
            elif font_size > body_avg * 1.3:
                return 'H2'
            elif font_size > body_avg * 1.1:
                return 'H3'
        
        return None
    
    def _is_heading_by_pattern(self, text: str) -> bool:
        """Check if text matches common heading patterns."""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        
        # Common heading patterns
        patterns = [
            r'^\d+\.?\s+[A-Z]',  # Numbered headings (1. Introduction)
            r'^[A-Z][^.!?]*$',   # All caps or title case without sentence ending
            r'^\d+\.\d+',        # Subsection numbering (1.1, 2.3)
            r'^(Chapter|Section|Part)\s+\d+',  # Explicit chapter/section
            r'^[IVX]+\.',        # Roman numerals
        ]
        
        for pattern in patterns:
            if re.match(pattern, text):
                return True
        
        # Check if it's likely a heading (short, capitalized, no sentence-ending punctuation)
        if (len(text) < 100 and 
            text[0].isupper() and 
            not text.endswith(('.', '!', '?')) and
            len(text.split()) < 10):
            return True
        
        return False
    
    def _post_process_sections(self, sections: List[DocumentSection]) -> List[DocumentSection]:
        """Post-process sections to improve quality."""
        processed_sections = []
        
        for section in sections:
            # Clean content
            content = re.sub(r'\s+', ' ', section.content.strip())
            content = re.sub(r'\n+', '\n', content)
            
            # Skip very short sections
            if len(content) < MIN_SECTION_LENGTH:
                continue
            
            # Update section
            processed_section = DocumentSection(
                title=section.title,
                content=content,
                page_number=section.page_number,
                level=section.level,
                font_size=section.font_size,
                document_name=section.document_name
            )
            
            processed_sections.append(processed_section)
        
        return processed_sections
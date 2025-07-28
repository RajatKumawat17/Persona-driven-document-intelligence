"""Configuration settings for the persona-driven document intelligence system."""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
INPUT_DIR = PROJECT_ROOT / "input"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODELS_DIR = PROJECT_ROOT / "models"

# Model configuration
MODEL_NAME = "all-mpnet-base-v2"  # 420MB, fast and efficient
# Alternative: "all-MiniLM-L6-v2 "  # 80MB, higher accuracy
MODEL_MAX_SIZE_MB = 1000  # As per challenge constraints

# Processing configuration
MAX_PROCESSING_TIME_SECONDS = 60
BATCH_SIZE = 32  # For embedding generation
MAX_SECTION_LENGTH = 2000  # Characters per section for processing
MIN_SECTION_LENGTH = 50  # Minimum characters to consider a section

# Relevance analysis
TOP_SECTIONS_COUNT = 10  # Number of top sections to include in output
SUBSECTION_SENTENCES_COUNT = 5  # Sentences to extract for refined text
SIMILARITY_THRESHOLD = 0.1  # Minimum similarity to consider relevant

# PDF processing
HEADING_FONT_SIZE_THRESHOLD = 1.2  # Multiplier for identifying headings
MIN_HEADING_LENGTH = 3  # Minimum characters for a heading
MAX_HEADING_LENGTH = 200  # Maximum characters for a heading

# Output formatting
OUTPUT_FILENAME = "analysis_results.json"
TIMESTAMP_FORMAT = "%Y-%m-%d %H:%M:%S"

# Logging
LOG_LEVEL = "INFO"

# Create directories if they don't exist
for directory in [INPUT_DIR, OUTPUT_DIR, MODELS_DIR]:
    directory.mkdir(exist_ok=True)
# Persona-Driven Document Intelligence

A sophisticated AI system that analyzes PDF documents and extracts the most relevant sections based on user persona and specific tasks.

## 🚀 Quick Start

### Docker Execution

```bash
# Build the Docker image
docker build --platform linux/amd64 -t persona-doc-intelligence:latest .

# Run the analysis
docker run --rm \
  -v $(pwd)/input:/app/input \
  -v $(pwd)/output:/app/output \
  --network none \
  persona-doc-intelligence:latest
```

### Input Requirements

Place the following in the `input/` directory:
- 3-10 PDF documents
- Configuration file (`config.json`) with:
```json
{
  "persona": "PhD Researcher in Computational Biology with expertise in machine learning",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies and benchmarks"
}
```

### Streamlit Web Interface

For interactive testing and development:

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run streamlit_app.py
```

## 🏗️ System Architecture

### Core Components

1. **PDF Processor** (`src/pdf_processor.py`)
   - Intelligent heading detection using multiple heuristics
   - Structure-aware text extraction
   - Robust handling of various PDF formats

2. **Embedding Engine** (`src/embedding_engine.py`)
   - Semantic vector generation using sentence-transformers
   - Context-rich query embedding creation
   - Efficient batch processing for scalability

3. **Relevance Ranker** (`src/relevance_ranker.py`)
   - Multi-factor relevance scoring
   - Document diversity analysis
   - Context-aware section ranking

4. **Extractive Summarizer** (`src/summarizer.py`)
   - Smart sentence selection algorithms
   - Position and keyword-aware scoring
   - Maintains content coherence

5. **Output Formatter** (`src/output_formatter.py`)
   - Challenge-compliant JSON formatting
   - Multi-format export capabilities
   - Comprehensive validation

## 🔧 Technical Specifications

### Model & Performance
- **Embedding Model**: `all-MiniLM-L6-v2` (80MB)
- **Architecture**: CPU-optimized, offline-capable
- **Processing Time**: <60 seconds for 3-5 documents
- **Memory Footprint**: <2GB RAM usage
- **Platform**: AMD64 (x86_64) compatible

### Key Features
- **Intelligent PDF Parsing**: Advanced heading detection beyond font-size heuristics
- **Semantic Understanding**: Deep contextual analysis using state-of-the-art embeddings  
- **Persona Integration**: User context deeply embedded in relevance scoring
- **Quality Summarization**: Multi-factor extractive summarization with diversity
- **Production Ready**: Robust error handling, logging, and monitoring

## 📊 Output Format

The system generates JSON output with the following structure:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "User persona description",
    "job_to_be_done": "Specific task description",
    "processing_timestamp": "2025-01-XX XX:XX:XX",
    "total_sections_analyzed": 25,
    "processing_stats": {...}
  },
  "extracted_sections": [
    {
      "document": "document_name",
      "page_number": 5,
      "section_title": "Section Title",
      "importance_rank": 1,
      "relevance_score": 0.8542,
      "section_level": "H2"
    }
  ],
  "subsection_analysis": [
    {
      "document": "document_name", 
      "page_number": 5,
      "refined_text": "Intelligent summary of the most relevant content...",
      "importance_rank": 1
    }
  ]
}
```

## 🎯 Sample Use Cases

### Academic Research
```json
{
  "persona": "PhD Researcher in Computational Biology with expertise in machine learning and drug discovery",
  "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
}
```

### Business Analysis  
```json
{
  "persona": "Investment Analyst with 5+ years experience in tech sector analysis and financial modeling",
  "job_to_be_done": "Analyze revenue trends, R&D investments, and market positioning strategies"
}
```

### Educational Support
```json
{
  "persona": "Undergraduate Chemistry Student preparing for organic chemistry exams",
  "job_to_be_done": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
}
```

## 🧪 Testing & Validation

### Running Tests
```bash
# Install test dependencies
pip install pytest

# Run unit tests
pytest tests/ -v

# Run integration tests
python -m pytest tests/test_components.py
```

### Performance Benchmarks
- **Small PDFs (10-20 pages)**: ~15-25 seconds
- **Medium PDFs (30-40 pages)**: ~35-45 seconds  
- **Large PDFs (45-50 pages)**: ~50-60 seconds
- **Memory Usage**: 1.2-1.8GB peak during processing

## 🛠️ Development Setup

### Prerequisites
- Python 3.9+
- Docker (for containerized execution)
- 8GB+ RAM recommended
- AMD64/x86_64 architecture

### Installation
```bash
# Clone repository
git clone <repository-url>
cd persona-document-intelligence

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download and cache model (optional - happens automatically)
python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

### Configuration
Modify `config/settings.py` to adjust:
- Model parameters and thresholds
- Processing limits and timeouts
- Output formatting options
- Logging levels

## 📁 Project Structure

```
persona-document-intelligence/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── Dockerfile                   # Container configuration
├── approach_explanation.md      # Technical approach details
├── main.py                      # Docker entry point
├── streamlit_app.py            # Web interface
├── config/
│   └── settings.py             # Configuration parameters
├── src/                        # Core application modules
│   ├── pdf_processor.py        # PDF parsing and extraction
│   ├── embedding_engine.py     # Semantic embedding generation
│   ├── relevance_ranker.py     # Section relevance analysis
│   ├── summarizer.py           # Extractive summarization
│   └── output_formatter.py     # Result formatting
├── models/                     # Local model cache (auto-created)
├── input/                      # Input PDFs and configuration
├── output/                     # Generated analysis results
├── sample_data/                # Test cases and examples
└── tests/                      # Unit and integration tests
```

## 🔍 Algorithm Details

### Relevance Scoring
The system uses a sophisticated multi-factor scoring algorithm:

1. **Semantic Similarity** (70% weight): Cosine similarity between query and section embeddings
2. **Position Importance** (15% weight): Beginning and end sections often contain key information
3. **Length Quality** (10% weight): Preference for medium-length, information-rich sections
4. **Keyword Presence** (5% weight): Boost for sections containing domain-relevant terms

### Summarization Strategy
The extractive summarizer employs:
- **Sentence-level analysis**: Individual sentence relevance scoring
- **Diversity optimization**: Prevents selection of redundant adjacent sentences
- **Context preservation**: Maintains original order for coherent summaries
- **Adaptive length**: Adjusts summary size based on content richness

## 🚀 Performance Optimizations

- **Batch Processing**: Efficient embedding generation in optimized batches
- **Memory Management**: Careful resource allocation for large document sets
- **CPU Optimization**: Leverages optimized linear algebra libraries
- **Caching**: Model and intermediate result caching for faster processing
- **Lazy Loading**: Components loaded only when needed

## 🐛 Troubleshooting

### Common Issues

**Model Download Fails**
```bash
# Manual model download
python -c "
from sentence_transformers import SentenceTransformer
import os
os.makedirs('models', exist_ok=True)
model = SentenceTransformer('all-MiniLM-L6-v2')
model.save('models/all-MiniLM-L6-v2')
"
```

**Memory Issues**
- Reduce `BATCH_SIZE` in `config/settings.py`
- Process fewer documents simultaneously
- Increase system RAM or swap space

**PDF Processing Errors**
- Ensure PDFs are not password-protected
- Check PDF file integrity
- Verify adequate disk space for temporary files

**Docker Build Issues**
- Ensure Docker supports AMD64 platform
- Check internet connectivity during build
- Verify sufficient disk space

## 📈 Future Enhancements

- **Multi-language Support**: Extend to non-English documents
- **Advanced PDF Handling**: Better support for complex layouts and images  
- **Interactive Refinement**: User feedback integration for improved relevance
- **Batch API**: REST API for programmatic access
- **Cloud Deployment**: Scalable cloud-native architecture

## 📄 License & Attribution

This project was developed for the Adobe India Hackathon 2025. It demonstrates advanced document intelligence techniques using state-of-the-art NLP models and efficient processing algorithms.

### Key Technologies
- **sentence-transformers**: Semantic embedding generation
- **PyMuPDF**: PDF parsing and text extraction
- **Streamlit**: Interactive web interface
- **NumPy/SciPy**: Numerical computing and similarity calculations

## 👥 Support

For questions, issues, or contributions:
1. Check the troubleshooting section above
2. Review the approach explanation document
3. Examine the code documentation and comments
4. Test with the provided sample configurations

---

**Built with ❤️ for intelligent document analysis**
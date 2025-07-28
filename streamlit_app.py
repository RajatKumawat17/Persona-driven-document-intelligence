"""
Streamlit frontend for Persona-Driven Document Intelligence.
"""

import streamlit as st
import json
import time
import tempfile
import os
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Add src to path
import sys
sys.path.append('src')

from src.pdf_processor import PDFProcessor
from src.embedding_engine import EmbeddingEngine
from src.relevance_ranker import RelevanceRanker
from src.summarizer import ExtractiveSummarizer
from src.output_formatter import OutputFormatter

# Page configuration
st.set_page_config(
    page_title="Persona-Driven Document Intelligence",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #ff7f0e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #c3e6cb;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #f5c6cb;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'processing_stats' not in st.session_state:
    st.session_state.processing_stats = None

@st.cache_resource
def initialize_system():
    """Initialize the analysis system components."""
    try:
        pdf_processor = PDFProcessor()
        embedding_engine = EmbeddingEngine()
        
        # Load model
        if embedding_engine.load_model():
            relevance_ranker = RelevanceRanker(embedding_engine)
            summarizer = ExtractiveSummarizer(embedding_engine)
            output_formatter = OutputFormatter()
            
            return {
                'pdf_processor': pdf_processor,
                'embedding_engine': embedding_engine,
                'relevance_ranker': relevance_ranker,
                'summarizer': summarizer,
                'output_formatter': output_formatter
            }
        else:
            return None
    except Exception as e:
        st.error(f"Error initializing system: {str(e)}")
        return None

def process_uploaded_files(uploaded_files: List, persona: str, job_to_be_done: str, components: Dict):
    """Process uploaded PDF files and return analysis results."""
    try:
        with st.spinner("Processing documents..."):
            # Create temporary directory
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                
                # Save uploaded files
                pdf_paths = []
                for uploaded_file in uploaded_files:
                    file_path = temp_path / uploaded_file.name
                    with open(file_path, 'wb') as f:
                        f.write(uploaded_file.getvalue())
                    pdf_paths.append(str(file_path))
                
                # Initialize processing stats
                processing_stats = {}
                
                # Process documents
                start_time = time.time()
                all_sections = []
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, pdf_path in enumerate(pdf_paths):
                    status_text.text(f"Processing {Path(pdf_path).name}...")
                    sections = components['pdf_processor'].process_pdf(pdf_path)
                    all_sections.extend(sections)
                    progress_bar.progress((i + 1) / len(pdf_paths))
                
                processing_stats['pdf_processing_time'] = time.time() - start_time
                
                if not all_sections:
                    st.error("No sections extracted from documents")
                    return None, None
                
                # Analyze relevance
                status_text.text("Analyzing relevance...")
                start_time = time.time()
                ranked_sections = components['relevance_ranker'].rank_sections(
                    all_sections, persona, job_to_be_done
                )
                processing_stats['ranking_time'] = time.time() - start_time
                
                if not ranked_sections:
                    st.error("No relevant sections found")
                    return None, None
                
                # Generate summaries
                status_text.text("Generating summaries...")
                start_time = time.time()
                query_embedding = components['embedding_engine'].create_query_embedding(
                    persona, job_to_be_done
                )
                
                summaries = {}
                for i, ranked_section in enumerate(ranked_sections):
                    summary = components['summarizer'].summarize_section(
                        ranked_section.section.content, query_embedding
                    )
                    summaries[i] = summary
                
                processing_stats['summarization_time'] = time.time() - start_time
                
                # Format results
                input_documents = [uploaded_file.name for uploaded_file in uploaded_files]
                results = components['output_formatter'].format_results(
                    ranked_sections=ranked_sections,
                    summarized_sections=summaries,
                    input_documents=input_documents,
                    persona=persona,
                    job_to_be_done=job_to_be_done,
                    processing_stats=processing_stats
                )
                
                progress_bar.empty()
                status_text.empty()
                
                return results, processing_stats
                
    except Exception as e:
        st.error(f"Error processing files: {str(e)}")
        return None, None

def display_results(results: Dict[str, Any], stats: Dict[str, Any]):
    """Display analysis results in the Streamlit interface."""
    
    st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
    
    # Metadata
    metadata = results.get('metadata', {})
    
    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Documents Processed",
            value=len(metadata.get('input_documents', []))
        )
    
    with col2:
        st.metric(
            label="Sections Found",
            value=len(results.get('extracted_sections', []))
        )
    
    with col3:
        st.metric(
            label="Summaries Generated",
            value=len(results.get('subsection_analysis', []))
        )
    
    with col4:
        st.metric(
            label="Processing Time",
            value=f"{stats.get('total_time', 0):.2f}s"
        )
    
    # Processing timeline
    st.markdown('<div class="section-header">⏱️ Processing Timeline</div>', unsafe_allow_html=True)
    
    timeline_data = {
        'Stage': ['PDF Processing', 'Ranking', 'Summarization'],
        'Time (seconds)': [
            stats.get('pdf_processing_time', 0),
            stats.get('ranking_time', 0),
            stats.get('summarization_time', 0)
        ]
    }
    
    fig = px.bar(timeline_data, x='Stage', y='Time (seconds)', 
                title='Processing Time by Stage')
    st.plotly_chart(fig, use_container_width=True)
    
    # Top relevant sections
    st.markdown('<div class="section-header">🎯 Top Relevant Sections</div>', unsafe_allow_html=True)
    
    extracted_sections = results.get('extracted_sections', [])
    if extracted_sections:
        # Create DataFrame for better display
        sections_df = pd.DataFrame(extracted_sections)
        
        # Display as interactive table
        st.dataframe(
            sections_df[['importance_rank', 'section_title', 'document', 'page_number', 'relevance_score']],
            use_container_width=True
        )
        
        # Relevance score distribution
        fig_scores = px.histogram(
            sections_df, x='relevance_score', nbins=10,
            title='Relevance Score Distribution'
        )
        st.plotly_chart(fig_scores, use_container_width=True)
        
        # Document distribution
        doc_counts = sections_df['document'].value_counts()
        fig_docs = px.pie(
            values=doc_counts.values, 
            names=doc_counts.index,
            title='Sections Distribution by Document'
        )
        st.plotly_chart(fig_docs, use_container_width=True)
    
    # Detailed analysis
    st.markdown('<div class="section-header">📝 Detailed Analysis</div>', unsafe_allow_html=True)
    
    subsection_analysis = results.get('subsection_analysis', [])
    
    for i, analysis in enumerate(subsection_analysis[:5]):  # Show top 5
        with st.expander(f"Rank {analysis['importance_rank']}: {extracted_sections[i]['section_title']}"):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                st.write(f"**Document:** {analysis['document']}")
                st.write(f"**Page:** {analysis['page_number']}")
                st.write(f"**Relevance Score:** {extracted_sections[i].get('relevance_score', 'N/A'):.4f}")
            
            with col2:
                st.write("**Refined Summary:**")
                st.write(analysis['refined_text'])

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<div class="main-header">📄 Persona-Driven Document Intelligence</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    Welcome to the Persona-Driven Document Intelligence system! 
    Upload your PDF documents, define your persona and job-to-be-done, 
    and get AI-powered insights tailored to your specific needs.
    """)
    
    # Initialize system
    with st.spinner("Initializing AI models..."):
        components = initialize_system()
    
    if not components:
        st.error("Failed to initialize the system. Please check the logs and try again.")
        return
    
    st.success("✅ System initialized successfully!")
    
    # Sidebar for inputs
    with st.sidebar:
        st.markdown("## 📋 Configuration")
        
        # File uploader
        uploaded_files = st.file_uploader(
            "Upload PDF Documents",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload 3-10 related PDF documents"
        )
        
        # Persona input
        persona = st.text_area(
            "Persona Description",
            placeholder="e.g., PhD Researcher in Computational Biology with expertise in machine learning and drug discovery",
            height=100,
            help="Describe the user persona including role, expertise, and focus areas"
        )
        
        # Job-to-be-done input
        job_to_be_done = st.text_area(
            "Job to be Done",
            placeholder="e.g., Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
            height=100,
            help="Describe the specific task or goal"
        )
        
        # Sample configurations
        st.markdown("### 🎯 Sample Configurations")
        
        sample_configs = {
            "Academic Researcher": {
                "persona": "PhD Researcher in Computational Biology with expertise in machine learning and drug discovery",
                "job": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
            },
            "Investment Analyst": {
                "persona": "Investment Analyst with 5+ years experience in tech sector analysis and financial modeling",
                "job": "Analyze revenue trends, R&D investments, and market positioning strategies"
            },
            "Chemistry Student": {
                "persona": "Undergraduate Chemistry Student preparing for organic chemistry exams",
                "job": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
            }
        }
        
        selected_sample = st.selectbox("Choose a sample configuration:", 
                                      [""] + list(sample_configs.keys()))
        
        if selected_sample and st.button("Load Sample"):
            st.session_state.sample_persona = sample_configs[selected_sample]["persona"]
            st.session_state.sample_job = sample_configs[selected_sample]["job"]
            st.rerun()
        
        # Use sample values if loaded
        if 'sample_persona' in st.session_state:
            persona = st.session_state.sample_persona
            job_to_be_done = st.session_state.sample_job
        
        # Process button
        process_button = st.button(
            "🚀 Start Analysis",
            type="primary",
            disabled=not (uploaded_files and persona and job_to_be_done)
        )
    
    # Main content area
    if process_button:
        if len(uploaded_files) < 3 or len(uploaded_files) > 10:
            st.warning("Please upload 3-10 PDF documents as specified in the challenge.")
            return
        
        # Process files
        start_time = time.time()
        results, stats = process_uploaded_files(uploaded_files, persona, job_to_be_done, components)
        
        if results:
            # Add total time to stats
            stats['total_time'] = time.time() - start_time
            
            # Store in session state
            st.session_state.analysis_results = results
            st.session_state.processing_stats = stats
            
            st.markdown('<div class="success-message">✅ Analysis completed successfully!</div>', 
                       unsafe_allow_html=True)
            
            # Display results
            display_results(results, stats)
            
            # Download options
            st.markdown('<div class="section-header">💾 Download Results</div>', unsafe_allow_html=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # JSON download
                json_str = json.dumps(results, indent=2, ensure_ascii=False)
                st.download_button(
                    label="📄 Download JSON",
                    data=json_str,
                    file_name="analysis_results.json",
                    mime="application/json"
                )
            
            with col2:
                # Summary report download
                summary_report = components['output_formatter'].create_summary_report(results)
                st.download_button(
                    label="📋 Download Report",
                    data=summary_report,
                    file_name="summary_report.txt",
                    mime="text/plain"
                )
            
            with col3:
                # Compact JSON download
                compact_json = json.dumps(results, separators=(',', ':'), ensure_ascii=False)
                st.download_button(
                    label="📦 Download Compact JSON",
                    data=compact_json,
                    file_name="analysis_results_compact.json",
                    mime="application/json"
                )
        
        else:
            st.markdown('<div class="error-message">❌ Analysis failed. Please check your inputs and try again.</div>', 
                       unsafe_allow_html=True)
    
    # Display previous results if available
    elif st.session_state.analysis_results:
        st.info("Showing previous analysis results. Upload new files to run a new analysis.")
        display_results(st.session_state.analysis_results, st.session_state.processing_stats)
    
    else:
        # Instructions
        st.markdown("""
        ## 🚀 Getting Started
        
        1. **Upload Documents**: Select 3-10 related PDF files using the sidebar
        2. **Define Persona**: Describe the user role, expertise, and focus areas
        3. **Specify Job**: Clearly state what task needs to be accomplished
        4. **Run Analysis**: Click "Start Analysis" to process your documents
        
        ## 🎯 Example Use Cases
        
        - **Academic Research**: Literature review preparation, methodology analysis
        - **Business Analysis**: Market research, competitive analysis, financial assessment
        - **Educational Support**: Study material identification, concept extraction
        - **Legal Research**: Case analysis, regulation review, precedent identification
        
        ## 📊 What You'll Get
        
        - **Ranked Sections**: Most relevant document sections based on your needs
        - **Smart Summaries**: Extractive summaries focusing on key information
        - **Visual Analytics**: Charts and insights about document coverage
        - **Multiple Formats**: JSON, text reports, and compact formats for download
        """)

if __name__ == "__main__":
    main()
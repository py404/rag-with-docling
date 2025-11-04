"""
Streamlit PDF RAG Application using Docling
Allows users to upload PDFs or provide URLs, convert with Docling, and query content
"""

import streamlit as st
from pathlib import Path
import tempfile
import time
from typing import List, Tuple

# Docling imports
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker

# Vector store and embeddings
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer
import numpy as np

# Page configuration
st.set_page_config(
    page_title="PDF RAG with Docling",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'collection' not in st.session_state:
    st.session_state.collection = None
if 'doc' not in st.session_state:
    st.session_state.doc = None
if 'markdown_content' not in st.session_state:
    st.session_state.markdown_content = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = []
if 'model' not in st.session_state:
    st.session_state.model = None
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False


@st.cache_resource
def load_embedding_model():
    """Load the sentence transformer model (cached)"""
    return SentenceTransformer('all-MiniLM-L6-v2')


def initialize_docling_converter(use_ocr: bool = True, use_tables: bool = True):
    """Initialize Docling document converter with specified options"""
    
    # Configure accelerator (MPS for Mac, CPU otherwise)
    accelerator_options = AcceleratorOptions(
        num_threads=8, 
        device=AcceleratorDevice.MPS  # Change to CPU if not on Mac
    )
    
    # Configure pipeline options
    pipeline_options = PdfPipelineOptions()
    pipeline_options.accelerator_options = accelerator_options
    pipeline_options.do_ocr = use_ocr
    pipeline_options.do_table_structure = use_tables
    if use_tables:
        pipeline_options.table_structure_options.do_cell_matching = True
    
    # Create converter
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=pipeline_options,
            )
        }
    )
    
    return converter


def setup_milvus_collection(collection_name: str = "docling_rag"):
    """Setup Milvus collection for vector storage"""
    
    # Connect to Milvus Lite
    connections.connect(uri='app.db')
    
    # Define schema
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
        FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=384)
    ]
    schema = CollectionSchema(fields, "PDF RAG Collection")
    
    # Drop existing collection if it exists
    if utility.has_collection(collection_name):
        utility.drop_collection(collection_name)
    
    # Create new collection
    collection = Collection(collection_name, schema)
    
    # Create index
    index_params = {
        "index_type": "IVF_FLAT",
        "metric_type": "L2",
        "params": {"nlist": 1024},
    }
    collection.create_index("vector", index_params)
    
    return collection


def process_pdf(source, converter, is_url: bool = False):
    """Process PDF from file or URL using Docling"""
    
    with st.spinner("🔄 Converting PDF with Docling..."):
        start_time = time.time()
        
        if is_url:
            conversion_result = converter.convert(source)
        else:
            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(source.read())
                tmp_path = tmp_file.name
            
            conversion_result = converter.convert(tmp_path)
        
        doc = conversion_result.document
        conversion_time = time.time() - start_time
        
        # Export to markdown
        markdown_content = doc.export_to_markdown()
        
        # Get confidence scores
        mean_grade = conversion_result.confidence.mean_grade
        low_grade = conversion_result.confidence.low_grade
        
        return doc, markdown_content, conversion_time, mean_grade, low_grade


def chunk_document(doc):
    """Chunk document using Docling's HybridChunker"""
    
    with st.spinner("✂️ Chunking document..."):
        chunker = HybridChunker()
        chunks_iterator = chunker.chunk(doc)
        chunks = list(chunks_iterator)
        
        return chunks


def create_embeddings(text: str, model) -> List[float]:
    """Create embeddings for text"""
    return model.encode(text).tolist()


def ingest_to_milvus(chunks, collection, model):
    """Ingest chunks and embeddings into Milvus"""
    
    with st.spinner("📥 Ingesting into vector database..."):
        # Extract texts and create embeddings
        texts = [chunk.text for chunk in chunks]
        embeddings = [create_embeddings(chunk.text, model) for chunk in chunks]
        embeddings_np = np.array(embeddings, dtype=np.float32)
        
        # Insert into Milvus
        entities = [texts, embeddings_np]
        collection.insert(entities)
        
        # Load collection for search
        collection.load()
        
        return len(chunks)


def search_similar_chunks(query: str, collection, model, top_k: int = 3) -> List[Tuple[str, float]]:
    """Search for similar chunks in the vector database"""
    
    # Create query embedding
    query_embedding = create_embeddings(query, model)
    
    # Perform search
    search_params = {"metric_type": "L2", "params": {"nprobe": 10}}
    results = collection.search([query_embedding], "vector", search_params, limit=top_k)
    
    # Extract results
    retrieved_chunks = []
    for hits in results:
        for hit in hits:
            result_entity = collection.query(
                expr=f"id == {hit.id}", 
                output_fields=["text"]
            )
            if result_entity:
                retrieved_chunks.append((result_entity[0]['text'], hit.score))
    
    return retrieved_chunks


# Main UI
st.title("📚 PDF RAG with Docling")
st.markdown("Upload a PDF or provide a URL to create a searchable knowledge base")

# Sidebar configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    
    use_ocr = st.checkbox("Enable OCR", value=True, help="Extract text from scanned PDFs")
    use_tables = st.checkbox("Extract Table Structure", value=True, help="Preserve table structure")
    top_k = st.slider("Number of results", min_value=1, max_value=10, value=3)
    
    st.divider()
    
    if st.button("🗑️ Clear Database", type="secondary"):
        if st.session_state.collection:
            st.session_state.collection = None
            st.session_state.processing_complete = False
            st.session_state.chunks = []
            st.success("Database cleared!")
            st.rerun()

# Main content area
tab1, tab2, tab3 = st.tabs(["📄 Upload & Process", "📝 Document View", "🔍 Query"])

with tab1:
    st.header("Step 1: Provide PDF Source")
    
    source_type = st.radio("Choose source type:", ["Upload File", "Provide URL"])
    
    pdf_source = None
    is_url = False
    
    if source_type == "Upload File":
        pdf_source = st.file_uploader("Upload PDF file", type=['pdf'])
    else:
        pdf_url = st.text_input("Enter PDF URL", placeholder="https://example.com/document.pdf")
        if pdf_url:
            pdf_source = pdf_url
            is_url = True
    
    if pdf_source:
        if st.button("🚀 Process PDF", type="primary"):
            try:
                # Load embedding model
                if st.session_state.model is None:
                    with st.spinner("Loading embedding model..."):
                        st.session_state.model = load_embedding_model()
                
                # Initialize converter
                converter = initialize_docling_converter(use_ocr, use_tables)
                
                # Process PDF
                doc, markdown, conv_time, mean_grade, low_grade = process_pdf(
                    pdf_source, converter, is_url
                )
                st.session_state.doc = doc
                st.session_state.markdown_content = markdown
                
                # Display conversion stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Conversion Time", f"{conv_time:.2f}s")
                with col2:
                    # Convert enum to string
                    st.metric("Mean Quality", str(mean_grade))
                with col3:
                    # Convert enum to string
                    st.metric("Low Quality", str(low_grade))
                
                # Chunk document
                chunks = chunk_document(doc)
                st.session_state.chunks = chunks
                st.success(f"✅ Created {len(chunks)} chunks")
                
                # Setup Milvus and ingest
                collection = setup_milvus_collection()
                st.session_state.collection = collection
                
                num_ingested = ingest_to_milvus(chunks, collection, st.session_state.model)
                st.success(f"✅ Ingested {num_ingested} chunks into vector database")
                
                st.session_state.processing_complete = True
                
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Error processing PDF: {str(e)}")
                st.exception(e)

with tab2:
    st.header("Document Content")
    
    if st.session_state.markdown_content:
        st.markdown("### Markdown Export")
        
        # Download button
        st.download_button(
            label="📥 Download Markdown",
            data=st.session_state.markdown_content,
            file_name="document.md",
            mime="text/markdown"
        )
        
        # Display markdown in expander
        with st.expander("View Markdown Content", expanded=False):
            st.markdown(st.session_state.markdown_content)
        
        # Show chunks
        if st.session_state.chunks:
            st.markdown(f"### Document Chunks ({len(st.session_state.chunks)} total)")
            
            chunk_to_show = st.number_input(
                "Select chunk to view", 
                min_value=1, 
                max_value=len(st.session_state.chunks),
                value=1
            )
            
            if chunk_to_show:
                chunk = st.session_state.chunks[chunk_to_show - 1]
                st.text_area(
                    f"Chunk {chunk_to_show}",
                    value=chunk.text,
                    height=300
                )
    else:
        st.info("👈 Process a PDF first to view its content")

with tab3:
    st.header("Query Your Document")
    
    if st.session_state.processing_complete:
        query = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?"
        )
        
        if query:
            if st.button("🔍 Search", type="primary"):
                try:
                    results = search_similar_chunks(
                        query, 
                        st.session_state.collection,
                        st.session_state.model,
                        top_k
                    )
                    
                    st.markdown("### 📊 Search Results")
                    
                    for i, (text, score) in enumerate(results, 1):
                        with st.expander(f"Result {i} (Score: {score:.4f})", expanded=(i==1)):
                            st.markdown(text)
                    
                    # Show context for RAG
                    st.markdown("### 🤖 Context for LLM")
                    context = "\n\n---\n\n".join([text for text, _ in results])
                    st.text_area(
                        "Combined context (copy this to your LLM):",
                        value=context,
                        height=300
                    )
                    
                except Exception as e:
                    st.error(f"❌ Error during search: {str(e)}")
        
        # Query history
        if 'query_history' not in st.session_state:
            st.session_state.query_history = []
        
        if query and query not in st.session_state.query_history:
            st.session_state.query_history.append(query)
        
        if st.session_state.query_history:
            st.markdown("### 📜 Recent Queries")
            for q in reversed(st.session_state.query_history[-5:]):
                st.text(f"• {q}")
    else:
        st.info("👈 Process a PDF first before querying")

# Footer
st.divider()
st.markdown("""
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit, Docling, and Milvus | 
    <a href='https://github.com/docling-project/docling'>Docling Docs</a>
    </p>
</div>
""", unsafe_allow_html=True)

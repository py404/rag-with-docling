"""
Main Streamlit application.
"""

import streamlit as st
from langchain_text_splitters import RecursiveCharacterTextSplitter
from loguru import logger
from models import AppSettings, DocumentMetadata, SearchResult
from services import (
    DoclingService,
    DocumentProcessingError,
    EmbeddingService,
    VectorStoreError,
    VectorStoreService,
    create_services,
)

# Configure logger
logger.add("app.log", rotation="10 MB", retention="7 days", level="INFO")

# Initialize settings
settings = AppSettings()

# Page config
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
    initial_sidebar_state="expanded",
)

# Session state
if "services_initialized" not in st.session_state:
    st.session_state.services_initialized = False
if "docling_service" not in st.session_state:
    st.session_state.docling_service = None
if "embedding_service" not in st.session_state:
    st.session_state.embedding_service = None
if "vector_store_service" not in st.session_state:
    st.session_state.vector_store_service = None
if "document_metadata" not in st.session_state:
    st.session_state.document_metadata = None
if "markdown_content" not in st.session_state:
    st.session_state.markdown_content = None
if "chunks" not in st.session_state:
    st.session_state.chunks = []
if "processing_complete" not in st.session_state:
    st.session_state.processing_complete = False
if "query_history" not in st.session_state:
    st.session_state.query_history = []


def initialize_services() -> None:
    """Initialize services."""
    try:
        with st.spinner("🔧 Initializing services..."):
            docling_service, embedding_service, vector_store_service = create_services(settings)

            st.session_state.docling_service = docling_service
            st.session_state.embedding_service = embedding_service
            st.session_state.vector_store_service = vector_store_service
            st.session_state.services_initialized = True

            logger.info("Services initialized")
            st.success("✅ Services ready!")
    except Exception as e:
        logger.error(f"Init failed: {e}")
        st.error(f"❌ Init failed: {e}")
        st.stop()


def process_pdf_document(pdf_source: any, is_url: bool = False) -> None:  # noqa: ANN401
    """Process PDF document."""
    try:
        if not st.session_state.services_initialized:
            initialize_services()

        docling_service: DoclingService = st.session_state.docling_service
        embedding_service: EmbeddingService = st.session_state.embedding_service
        vector_store_service: VectorStoreService = st.session_state.vector_store_service

        # Process PDF
        with st.spinner("🔄 Processing PDF..."):
            markdown_content, metadata = docling_service.process_pdf(pdf_source, is_url)
            st.session_state.markdown_content = markdown_content
            st.session_state.document_metadata = metadata

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time", f"{metadata.processing_time:.2f}s")
        with col2:
            st.metric("Quality", metadata.mean_quality)
        with col3:
            st.metric("Tables", metadata.table_count)

        # Chunk document
        with st.spinner("✂️ Chunking..."):
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)

            class SimpleChunk:
                def __init__(self, text: str) -> None:
                    self.text = text

            text_chunks = text_splitter.split_text(markdown_content)
            chunks = [SimpleChunk(text) for text in text_chunks]
            st.session_state.chunks = chunks
            metadata.chunk_count = len(chunks)

        st.success(f"✅ Created {len(chunks)} chunks")

        # Generate embeddings
        with st.spinner("🔢 Generating embeddings..."):
            texts = [chunk.text for chunk in chunks]
            embeddings = embedding_service.create_embeddings_batch(texts)

        st.success(f"✅ Generated {len(embeddings)} embeddings")

        # Ingest
        with st.spinner("📥 Ingesting..."):
            vector_store_service.setup_collection()
            num_ingested = vector_store_service.ingest_documents(chunks, embeddings)

        st.success(f"✅ Ingested {num_ingested} chunks")
        st.session_state.processing_complete = True
        st.balloons()

    except DocumentProcessingError as e:
        logger.error(f"Processing error: {e}")
        st.error(f"❌ Processing failed: {e}")
    except VectorStoreError as e:
        logger.error(f"Vector store error: {e}")
        st.error(f"❌ Vector store failed: {e}")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        st.error(f"❌ Error: {e}")
        st.exception(e)


def perform_search(query: str, top_k: int) -> list[SearchResult]:
    """Perform search."""
    try:
        embedding_service: EmbeddingService = st.session_state.embedding_service
        vector_store_service: VectorStoreService = st.session_state.vector_store_service

        query_embedding = embedding_service.create_embedding(query)
        results = vector_store_service.search(query_embedding, top_k)

        if query not in st.session_state.query_history:
            st.session_state.query_history.append(query)

        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        st.error(f"❌ Search failed: {e}")
        return []


# Main UI
st.title(f"{settings.app_icon} {settings.app_title}")
st.markdown("Upload a PDF or provide a URL to create a searchable knowledge base")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    use_ocr = st.checkbox("Enable OCR", value=settings.processing.use_ocr)
    use_tables = st.checkbox("Extract Tables", value=settings.processing.use_table_structure)
    top_k = st.slider("Results", min_value=1, max_value=20, value=settings.search.top_k)

    settings.processing.use_ocr = use_ocr
    settings.processing.use_table_structure = use_tables
    settings.search.top_k = top_k

    st.divider()

    st.subheader("📊 Status")
    if st.session_state.services_initialized:
        st.success("Services: ✅")
    else:
        st.warning("Services: ⏳")

    if st.session_state.processing_complete:
        st.success("Document: ✅")
    else:
        st.info("Document: ⏳")

    st.divider()

    if st.button("🗑️ Clear", type="secondary"):
        st.session_state.processing_complete = False
        st.session_state.chunks = []
        st.session_state.document_metadata = None
        st.session_state.markdown_content = None
        st.success("Cleared!")
        st.rerun()

# Tabs
tab1, tab2, tab3 = st.tabs(["📄 Upload", "📝 Document", "🔍 Query"])

with tab1:
    st.header("Upload PDF")

    source_type = st.radio("Source:", ["Upload File", "Provide URL"])

    pdf_source = None
    is_url = False

    if source_type == "Upload File":
        pdf_source = st.file_uploader("Upload PDF", type=["pdf"])
    else:
        pdf_url = st.text_input("PDF URL", placeholder="https://example.com/doc.pdf")
        if pdf_url:
            pdf_source = pdf_url
            is_url = True

    if pdf_source and st.button("🚀 Process", type="primary"):
        process_pdf_document(pdf_source, is_url)

with tab2:
    st.header("Document")

    if st.session_state.markdown_content and st.session_state.document_metadata:
        metadata: DocumentMetadata = st.session_state.document_metadata

        st.info(
            f"📄 {metadata.filename} | {metadata.char_count:,} chars | {metadata.chunk_count} chunks"
        )

        st.download_button(
            "📥 Download",
            data=st.session_state.markdown_content,
            file_name=f"{metadata.filename}.md",
            mime="text/markdown",
        )

        with st.expander("View Content", expanded=False):
            st.markdown(st.session_state.markdown_content)

        if st.session_state.chunks:
            st.markdown(f"### Chunks ({len(st.session_state.chunks)})")

            chunk_num = st.number_input(
                "Select chunk",
                min_value=1,
                max_value=len(st.session_state.chunks),
                value=1,
            )

            if chunk_num:
                chunk = st.session_state.chunks[chunk_num - 1]
                st.text_area(f"Chunk {chunk_num}", value=chunk.text, height=300)
    else:
        st.info("👈 Process a PDF first")

with tab3:
    st.header("Query Your Document")

    if st.session_state.processing_complete:
        query = st.text_input("Enter your question:", placeholder="What is this document about?")

        if query:
            if st.button("🔍 Search", type="primary"):
                results = perform_search(query, top_k)

                if results:
                    st.markdown("### 📊 Search Results")

                    for i, result in enumerate(results, 1):
                        with st.expander(
                            f"Result {i} (Score: {result.score:.4f})", expanded=(i == 1)
                        ):
                            st.markdown(result.text)

                    # Show context for RAG
                    st.markdown("### 🤖 Context for LLM")
                    context = "\n\n---\n\n".join([r.text for r in results])
                    st.text_area(
                        "Combined context (copy this to your LLM):", value=context, height=300
                    )

        # Query history
        if st.session_state.query_history:
            st.markdown("### 📜 Recent Queries")
            for q in reversed(st.session_state.query_history[-5:]):
                st.text(f"• {q}")
    else:
        st.info("👈 Process a PDF first before querying")

# Footer
st.divider()
st.markdown(
    """
<div style='text-align: center; color: gray;'>
    <p>Built with Streamlit, Docling, and Milvus | 
    <a href='https://github.com/docling-project/docling'>Docling Docs</a>
    </p>
</div>
""",
    unsafe_allow_html=True,
)

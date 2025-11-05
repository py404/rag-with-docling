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
    VisualGroundingService,
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
if "docling_document" not in st.session_state:
    st.session_state.docling_document = None
if "doc_store" not in st.session_state:
    st.session_state.doc_store = {}
if "last_search_results" not in st.session_state:
    st.session_state.last_search_results = []
if "last_query" not in st.session_state:
    st.session_state.last_query = ""


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
            markdown_content, metadata, dl_doc = docling_service.process_pdf(pdf_source, is_url)
            st.session_state.markdown_content = markdown_content
            st.session_state.document_metadata = metadata
            st.session_state.docling_document = dl_doc

        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Time", f"{metadata.processing_time:.2f}s")
        with col2:
            st.metric("Quality", metadata.mean_quality)
        with col3:
            st.metric("Tables", metadata.table_count)

        # Chunk document
        with st.spinner("✂️ Chunking with provenance..."):
            # Use Docling's HybridChunker to preserve provenance information
            from docling.chunking import HybridChunker
            
            chunker = HybridChunker()
            chunk_iter = chunker.chunk(dl_doc)
            
            # Convert to list and extract text and metadata
            chunks_with_meta = []
            for chunk in chunk_iter:
                chunks_with_meta.append({
                    'text': chunk.text,
                    'meta': chunk.meta.export_json_dict() if hasattr(chunk, 'meta') else {}
                })
            
            st.session_state.chunks = chunks_with_meta
            metadata.chunk_count = len(chunks_with_meta)

        st.success(f"✅ Created {len(chunks_with_meta)} chunks with provenance")

        # Generate embeddings
        with st.spinner("🔢 Generating embeddings..."):
            texts = [chunk['text'] for chunk in chunks_with_meta]
            embeddings = embedding_service.create_embeddings_batch(texts)

        st.success(f"✅ Generated {len(embeddings)} embeddings")

        # Ingest
        with st.spinner("📥 Ingesting..."):
            vector_store_service.setup_collection()
            num_ingested = vector_store_service.ingest_documents(chunks_with_meta, embeddings)

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
        
        # Store results for visual grounding
        st.session_state.last_search_results = results
        st.session_state.last_query = query

        return results

    except Exception as e:
        logger.error(f"Search error: {e}")
        st.error(f"❌ Search failed: {e}")
        return []


def get_page_list(dl_doc):
    """Get a list of pages from DoclingDocument, handling both dict and list structures."""
    if not hasattr(dl_doc, 'pages'):
        return []
    
    if isinstance(dl_doc.pages, dict):
        # Return sorted list of page objects from dict
        return [dl_doc.pages[key] for key in sorted(dl_doc.pages.keys())]
    elif hasattr(dl_doc.pages, '__iter__'):
        # Return as list if it's iterable
        return list(dl_doc.pages)
    else:
        return []


def get_num_pages(dl_doc):
    """Get the number of pages in DoclingDocument."""
    if not hasattr(dl_doc, 'pages'):
        return 0
    return len(dl_doc.pages) if hasattr(dl_doc.pages, '__len__') else 0


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
        st.session_state.docling_document = None
        st.session_state.doc_store = {}
        st.session_state.last_search_results = []
        st.session_state.last_query = ""
        st.success("Cleared!")
        st.rerun()

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📄 Upload", "📝 Document", "🔍 Query", "🎯 Visual Grounding"])

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
                chunk_text = chunk['text'] if isinstance(chunk, dict) else chunk.text
                st.text_area(f"Chunk {chunk_num}", value=chunk_text, height=300)
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

with tab4:
    st.header("🎯 Visual Grounding")
    st.markdown(
        "See where your search results are located in the original document pages with highlighted bounding boxes."
    )

    if st.session_state.processing_complete and st.session_state.docling_document:
        dl_doc = st.session_state.docling_document
        
        # Check if document has pages with images
        has_images = False
        num_pages = get_num_pages(dl_doc)
        
        try:
            if num_pages > 0:
                pages_list = get_page_list(dl_doc)
                if pages_list:
                    first_page = pages_list[0]
                    has_images = (hasattr(first_page, 'image') and 
                                 first_page.image is not None and 
                                 hasattr(first_page.image, 'pil_image') and 
                                 first_page.image.pil_image is not None)
        except Exception as e:
            logger.warning(f"Could not check page images: {e}")
            has_images = False
        
        if not has_images:
            st.warning(
                "⚠️ **Page images are not available**. "
                "The document was processed but page images were not generated. "
                "Visual grounding requires page images to be enabled during document conversion."
            )
            with st.expander("🔧 How to fix this"):
                st.markdown("""
                The document converter needs to be configured with page image generation enabled.
                This is controlled in `services.py` in the `DoclingService._initialize_converter()` method.
                
                The current settings should include:
                ```python
                pipeline_options.generate_page_images = True
                pipeline_options.images_scale = 2.0
                ```
                
                If these settings are in place but images are still not available:
                1. Make sure you're using a recent version of Docling (>=2.0.0)
                2. Try reprocessing the document
                3. Check the logs for any errors during conversion
                """)
        elif st.session_state.last_search_results and st.session_state.last_query:
            st.info(f"📌 Showing results for: **{st.session_state.last_query}**")

            # Group results by page number for efficient rendering
            from services import VisualGroundingService
            
            results = st.session_state.last_search_results
            dl_doc = st.session_state.docling_document
            pages_list = get_page_list(dl_doc)
            
            # Extract all bounding boxes from search results grouped by page
            page_bboxes = {}  # page_num -> list of bboxes
            
            for idx, result in enumerate(results):
                logger.info(f"Processing result {idx}, has metadata: {bool(result.metadata)}")
                if result.metadata:
                    logger.debug(f"Result {idx} metadata keys: {result.metadata.keys()}")
                
                if result.metadata and 'doc_items' in result.metadata:
                    doc_items = result.metadata['doc_items']
                    logger.info(f"Result {idx} has {len(doc_items)} doc_items")
                    
                    for item_idx, item in enumerate(doc_items):
                        if 'prov' in item and item['prov']:
                            logger.debug(f"Result {idx}, item {item_idx} has {len(item['prov'])} provenance entries")
                            
                            for prov in item['prov']:
                                if 'page_no' in prov and 'bbox' in prov:
                                    page_no = prov['page_no']
                                    bbox = prov['bbox']
                                    
                                    logger.debug(f"Found bbox for page {page_no}: {bbox}")
                                    
                                    if page_no not in page_bboxes:
                                        page_bboxes[page_no] = []
                                    
                                    page_bboxes[page_no].append({
                                        'bbox': bbox,
                                        'result_idx': idx,
                                        'score': result.score
                                    })
                                else:
                                    logger.warning(f"Provenance missing page_no or bbox: {prov}")
                        else:
                            logger.debug(f"Result {idx}, item {item_idx} has no prov or empty prov")
                else:
                    logger.warning(f"Result {idx} has no doc_items in metadata")
            
            logger.info(f"Total pages with bboxes: {len(page_bboxes)}")
            for page_no, bboxes in page_bboxes.items():
                logger.info(f"Page {page_no}: {len(bboxes)} bboxes")
            
            if page_bboxes:
                st.success(f"✨ Found {len(page_bboxes)} pages with bounding boxes from {len(results)} search results")
                
                # Show results with bounding boxes
                st.markdown("### 🎯 Search Results with Visual Grounding")
                
                for i, result in enumerate(results, 1):
                    with st.expander(f"Result {i} (Score: {result.score:.4f})", expanded=True):
                        st.markdown(f"**Text:** {result.text[:300]}..." if len(result.text) > 300 else result.text)
                        
                        # Find which pages this result appears on
                        result_pages = []
                        if result.metadata and 'doc_items' in result.metadata:
                            for item in result.metadata['doc_items']:
                                if 'prov' in item and item['prov']:
                                    for prov in item['prov']:
                                        if 'page_no' in prov:
                                            page_no = prov['page_no']
                                            if page_no not in result_pages:
                                                result_pages.append(page_no)
                        
                        if result_pages:
                            st.markdown(f"**📄 Pages:** {', '.join([str(p+1) for p in result_pages])}")
                            
                            # Display images with bounding boxes for this result
                            for page_no in result_pages:
                                if page_no < len(pages_list):
                                    page = pages_list[page_no]
                                    
                                    # Get bboxes for this specific result on this page
                                    result_bboxes = [
                                        bbox_info for bbox_info in page_bboxes.get(page_no, [])
                                        if bbox_info['result_idx'] == i - 1
                                    ]
                                    
                                    if result_bboxes and hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                                        try:
                                            # Draw bounding boxes on the image
                                            img_with_boxes = VisualGroundingService.draw_bounding_boxes_on_image(
                                                dl_doc, page_no, result_bboxes
                                            )
                                            st.image(
                                                img_with_boxes,
                                                caption=f"Page {page_no + 1} - Highlighted region for Result {i}",
                                                use_container_width=True
                                            )
                                        except Exception as e:
                                            logger.error(f"Failed to draw bounding boxes on page {page_no}: {e}")
                                            st.error(f"Could not draw bounding boxes on page {page_no + 1}: {str(e)}")
                                            # Show the original image without boxes
                                            st.image(page.image.pil_image, caption=f"Page {page_no + 1} (without highlighting)", use_container_width=True)
                                            # Debug info
                                            with st.expander("🔍 Debug Info"):
                                                st.json(result_bboxes)
                                                st.code(f"Error: {type(e).__name__}: {str(e)}")
                                    elif hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                                        # No bboxes but has image
                                        st.image(page.image.pil_image, caption=f"Page {page_no + 1}", use_container_width=True)
                                        st.info("Bounding box data not available for this result")
                        else:
                            st.info("No page location information available for this result")
                
            else:
                st.warning(
                    "⚠️ No provenance information found in search results. "
                    "The chunks may not have been created with the HybridChunker, "
                    "or the document structure doesn't support provenance tracking."
                )
                
                # Fallback: show pages without bounding boxes
                st.markdown("### 📄 Document Pages")
                pages_list = get_page_list(dl_doc)
                num_pages = len(pages_list)
                
                page_num = st.selectbox(
                    "Select page to view:",
                    range(num_pages),
                    format_func=lambda x: f"Page {x + 1}",
                    key="vg_page_no_bbox"
                )
                
                if page_num is not None and page_num < num_pages:
                    page = pages_list[page_num]
                    if hasattr(page, 'image') and page.image and hasattr(page.image, 'pil_image'):
                        st.image(page.image.pil_image, caption=f"Page {page_num + 1}", use_container_width=True)

            # Show info about the feature
            with st.expander("ℹ️ About Visual Grounding"):
                st.markdown("""
                **Visual Grounding** shows you exactly where in the document your search results came from!
                
                This feature uses:
                - **Docling's HybridChunker**: Preserves provenance (location) information for each chunk
                - **Bounding Boxes**: Highlights the exact regions on document pages
                - **Metadata Storage**: Stores page numbers and coordinates with each chunk
                
                The blue boxes you see are drawn around the text regions that match your query.
                Multiple boxes on the same page indicate different parts of the same result or multiple results.
                """)

        else:
            # Show pages even without search results
            st.info("💡 You can browse document pages below. Perform a search in the Query tab to see visual grounding with search context.")
            
            # Display document pages
            pages_list = get_page_list(dl_doc)
            num_pages = len(pages_list)
            
            st.markdown(f"### 📄 Document Pages ({num_pages} pages)")
            
            # Page selector
            page_num = st.selectbox(
                "Select page to view:",
                range(num_pages),
                format_func=lambda x: f"Page {x + 1}",
                key="vg_page_selector_no_search"
            )
            
            if page_num is not None and page_num < num_pages:
                try:
                    page = pages_list[page_num]
                    
                    # Check if page has image attribute
                    if hasattr(page, 'image') and page.image is not None:
                        # Check if image has pil_image attribute
                        if hasattr(page.image, 'pil_image') and page.image.pil_image is not None:
                            st.image(
                                page.image.pil_image,
                                caption=f"Page {page_num + 1}",
                                use_container_width=True
                            )
                        else:
                            st.warning(f"⚠️ Page {page_num + 1} image object exists but pil_image is not available.")
                    else:
                        st.warning(f"⚠️ No image available for page {page_num + 1}")
                        
                    # Show page content summary
                    with st.expander(f"Page {page_num + 1} Content Summary"):
                        if hasattr(page, 'size'):
                            st.markdown(f"**Page dimensions**: {page.size.width} x {page.size.height}")
                        if hasattr(page, 'image'):
                            st.markdown(f"**Has image**: {page.image is not None}")
                            if page.image:
                                st.markdown(f"**Has PIL image**: {hasattr(page.image, 'pil_image') and page.image.pil_image is not None}")
                        
                except Exception as e:
                    st.error(f"❌ Error displaying page: {e}")
                    logger.error(f"Page display error for page {page_num}: {e}")
                    with st.expander("🔍 Debug Info"):
                        st.code(f"Error type: {type(e).__name__}\nError message: {str(e)}")
            
    elif not st.session_state.processing_complete:
        st.info("👈 Process a PDF first to enable visual grounding")
    else:
        st.warning("⚠️ Document does not have page images. Visual grounding requires page images.")

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

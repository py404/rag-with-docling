"""
Service layer for document processing, embeddings, and vector storage.
"""

import tempfile
import time
from pathlib import Path
from typing import BinaryIO
import json

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.datamodel.document import DoclingDocument
from loguru import logger
from models import (
    AppSettings,
    DocumentMetadata,
    EmbeddingConfig,
    ProcessingConfig,
    SearchResult,
    VectorStoreConfig,
)
from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, connections, utility
from sentence_transformers import SentenceTransformer
from PIL import Image, ImageDraw


class DocumentProcessingError(Exception):
    """Document processing error."""


class VectorStoreError(Exception):
    """Vector store error."""


class EmbeddingError(Exception):
    """Embedding error."""


class DoclingService:
    """PDF processing service."""

    def __init__(self, config: ProcessingConfig) -> None:
        """Initialize service."""
        self.config = config
        self.converter = self._initialize_converter()
        logger.info(f"Docling service initialized: {config.accelerator}")

    def _initialize_converter(self) -> DocumentConverter:
        """Initialize converter."""
        try:
            accelerator_map = {
                "mps": AcceleratorDevice.MPS,
                "cuda": AcceleratorDevice.CUDA,
                "cpu": AcceleratorDevice.CPU,
            }

            accelerator_options = AcceleratorOptions(
                num_threads=self.config.num_threads,
                device=accelerator_map[self.config.accelerator.value],
            )

            pipeline_options = PdfPipelineOptions()
            pipeline_options.accelerator_options = accelerator_options
            pipeline_options.do_ocr = self.config.use_ocr
            pipeline_options.do_table_structure = self.config.use_table_structure
            
            # Enable page image generation for visual grounding
            pipeline_options.generate_page_images = True
            pipeline_options.images_scale = 2.0

            if self.config.use_table_structure:
                pipeline_options.table_structure_options.do_cell_matching = True

            return DocumentConverter(
                format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
            )
        except Exception as e:
            logger.error(f"Converter init failed: {e}")
            raise DocumentProcessingError(f"Init failed: {e}") from e

    def process_pdf(
        self, source: str | BinaryIO, is_url: bool = False
    ) -> tuple[str, DocumentMetadata, DoclingDocument]:
        """Process PDF and return markdown with metadata and DoclingDocument."""
        start_time = time.time()

        try:
            if is_url:
                logger.info(f"Processing URL: {source}")
                conversion_result = self.converter.convert(source)
                source_path = str(source)
                filename = Path(source).name if isinstance(source, str) else "url_document.pdf"
            else:
                logger.info("Processing uploaded file")
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    if hasattr(source, "read"):
                        tmp_file.write(source.read())
                    else:
                        raise ValueError("Invalid source type")
                    tmp_path = tmp_file.name

                conversion_result = self.converter.convert(tmp_path)
                source_path = tmp_path
                filename = getattr(source, "name", "uploaded_document.pdf")

            doc = conversion_result.document
            markdown_content = doc.export_to_markdown()
            processing_time = time.time() - start_time

            metadata = DocumentMetadata(
                filename=filename,
                source=source_path,
                char_count=len(markdown_content),
                chunk_count=0,
                has_tables=len(doc.tables) > 0,
                table_count=len(doc.tables),
                processing_time=processing_time,
                mean_quality=str(conversion_result.confidence.mean_grade),
                low_quality=str(conversion_result.confidence.low_grade),
            )

            logger.info(f"Processed: {filename} ({processing_time:.2f}s)")
            return markdown_content, metadata, doc

        except Exception as e:
            logger.error(f"Processing failed: {e}")
            raise DocumentProcessingError(f"Failed: {e}") from e


class EmbeddingService:
    """Embedding generation service."""

    def __init__(self, config: EmbeddingConfig) -> None:
        """Initialize service."""
        self.config = config
        self.model = self._load_model()
        logger.info(f"Embedding service initialized: {config.model_name}")

    def _load_model(self) -> SentenceTransformer:
        """Load model."""
        try:
            return SentenceTransformer(self.config.model_name)
        except Exception as e:
            logger.error(f"Model load failed: {e}")
            raise EmbeddingError(f"Load failed: {e}") from e

    def create_embedding(self, text: str) -> list[float]:
        """Create embedding."""
        try:
            return self.model.encode(text).tolist()
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            raise EmbeddingError(f"Failed: {e}") from e

    def create_embeddings_batch(self, texts: list[str]) -> np.ndarray:
        """Create batch embeddings."""
        try:
            logger.info(f"Creating {len(texts)} embeddings")
            embeddings = [self.create_embedding(text) for text in texts]
            return np.array(embeddings, dtype=np.float32)
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise EmbeddingError(f"Batch failed: {e}") from e


class VectorStoreService:
    """Vector database service."""

    def __init__(self, config: VectorStoreConfig, embedding_dim: int = 384) -> None:
        """Initialize service."""
        self.config = config
        self.embedding_dim = embedding_dim
        self.collection: Collection | None = None
        logger.info(f"Vector store initialized: {config.collection_name}")

    def setup_collection(self) -> Collection:
        """Setup collection."""
        try:
            connections.connect(uri=self.config.uri)
            logger.info(f"Connected to Milvus: {self.config.uri}")

            fields = [
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, "PDF RAG Collection with Metadata")

            if utility.has_collection(self.config.collection_name):
                logger.warning(f"Dropping collection: {self.config.collection_name}")
                utility.drop_collection(self.config.collection_name)

            collection = Collection(self.config.collection_name, schema)

            index_params = {
                "index_type": self.config.index_type,
                "metric_type": self.config.metric_type,
                "params": {"nlist": self.config.nlist},
            }
            collection.create_index("vector", index_params)
            logger.info(f"Created collection with index: {self.config.index_type}")

            self.collection = collection
            return collection

        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise VectorStoreError(f"Setup failed: {e}") from e

    def ingest_documents(self, chunks: list[any], embeddings: np.ndarray) -> int:  # noqa: ANN401
        """Ingest documents with metadata."""
        if self.collection is None:
            raise VectorStoreError("Collection not initialized")

        try:
            logger.info(f"Ingesting {len(chunks)} chunks")
            
            # Handle both old format (SimpleChunk) and new format (dict with metadata)
            texts = []
            metadata_list = []
            
            for chunk in chunks:
                if isinstance(chunk, dict):
                    texts.append(chunk['text'])
                    # Serialize metadata as JSON string
                    metadata_list.append(json.dumps(chunk.get('meta', {})))
                else:
                    # Old format compatibility
                    texts.append(chunk.text if hasattr(chunk, 'text') else str(chunk))
                    metadata_list.append(json.dumps({}))
            
            entities = [texts, metadata_list, embeddings]
            self.collection.insert(entities)
            self.collection.load()
            logger.info(f"Ingested {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise VectorStoreError(f"Ingestion failed: {e}") from e

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """Search for similar chunks with metadata."""
        if self.collection is None:
            raise VectorStoreError("Collection not initialized")

        try:
            logger.info(f"Searching top {top_k}")

            search_params = {
                "metric_type": self.config.metric_type,
                "params": {"nprobe": self.config.nprobe},
            }
            results = self.collection.search(
                [query_embedding], "vector", search_params, limit=top_k
            )

            search_results = []
            for hits in results:
                for hit in hits:
                    result_entity = self.collection.query(
                        expr=f"id == {hit.id}", output_fields=["text", "metadata"]
                    )
                    if result_entity:
                        # Parse metadata from JSON string
                        metadata = {}
                        try:
                            metadata = json.loads(result_entity[0].get("metadata", "{}"))
                        except json.JSONDecodeError:
                            logger.warning(f"Failed to parse metadata for chunk {hit.id}")
                        
                        search_results.append(
                            SearchResult(
                                text=result_entity[0]["text"],
                                score=float(hit.score),
                                chunk_id=int(hit.id),
                                metadata=metadata,
                            )
                        )

            logger.info(f"Found {len(search_results)} results")
            return search_results

        except Exception as e:
            logger.error(f"Search failed: {e}")
            raise VectorStoreError(f"Search failed: {e}") from e


def create_services(
    settings: AppSettings,
) -> tuple[DoclingService, EmbeddingService, VectorStoreService]:
    """Create all services."""
    docling_service = DoclingService(settings.processing)
    embedding_service = EmbeddingService(settings.embedding)
    vector_store_service = VectorStoreService(settings.vector_store, settings.embedding.dimension)

    return docling_service, embedding_service, vector_store_service


class VisualGroundingService:
    """Service for visual grounding with bounding boxes."""
    
    @staticmethod
    def draw_bounding_boxes_on_image(
        dl_doc: DoclingDocument,
        page_no: int,
        bboxes: list[dict]
    ) -> Image.Image:
        """
        Draw bounding boxes on page image.
        
        Args:
            dl_doc: DoclingDocument with page images
            page_no: Page number (0-indexed)
            bboxes: List of bounding box dictionaries with coordinates
            
        Returns:
            PIL Image with bounding boxes drawn
        """
        try:
            logger.info(f"Attempting to draw {len(bboxes)} bounding boxes on page {page_no}")
            
            # Get page using helper method to handle dict/list structure
            if isinstance(dl_doc.pages, dict):
                page_keys = sorted(dl_doc.pages.keys())
                if page_no >= len(page_keys):
                    raise ValueError(f"Page {page_no} not found in document with {len(page_keys)} pages")
                page = dl_doc.pages[page_keys[page_no]]
            else:
                pages_list = list(dl_doc.pages)
                if page_no >= len(pages_list):
                    raise ValueError(f"Page {page_no} not found in document with {len(pages_list)} pages")
                page = pages_list[page_no]
            
            if not hasattr(page, 'image') or page.image is None:
                raise ValueError(f"Page {page_no} does not have an image")
            
            if not hasattr(page.image, 'pil_image') or page.image.pil_image is None:
                raise ValueError(f"Page {page_no} image does not have pil_image")
            
            img = page.image.pil_image.copy()
            logger.info(f"Got image for page {page_no}: {img.size}")
            
            draw = ImageDraw.Draw(img)
            thickness = 4
            padding = thickness + 2
            
            boxes_drawn = 0
            for idx, bbox_dict in enumerate(bboxes):
                logger.debug(f"Processing bbox {idx}: {bbox_dict}")
                
                # Convert bbox coordinates to pixel coordinates
                bbox_coords = bbox_dict.get("bbox", {})
                if not bbox_coords:
                    logger.warning(f"Bbox {idx} has no 'bbox' key")
                    continue
                
                # Check coordinate origin
                coord_origin = bbox_coords.get("coord_origin", "TOPLEFT")
                
                # Get bbox coordinates - Docling uses different coordinate systems
                if 'l' in bbox_coords and 'r' in bbox_coords and 't' in bbox_coords and 'b' in bbox_coords:
                    l = float(bbox_coords.get("l", 0))
                    r = float(bbox_coords.get("r", 1))
                    t = float(bbox_coords.get("t", 0))
                    b = float(bbox_coords.get("b", 1))
                    
                    # Check if coordinates are normalized (0-1 range) or absolute (in points/pixels)
                    # If any value is > 1, assume absolute coordinates
                    if l <= 1 and r <= 1 and t <= 1 and b <= 1:
                        # Normalized coordinates (0-1 range)
                        logger.debug(f"Using normalized coordinates")
                        
                        # Ensure proper ordering
                        x_min = min(l, r)
                        x_max = max(l, r)
                        y_min = min(t, b)
                        y_max = max(t, b)
                        
                        # Scale to image size
                        l_px = max(0, int(x_min * img.width) - padding)
                        r_px = min(img.width, int(x_max * img.width) + padding)
                        t_px = max(0, int(y_min * img.height) - padding)
                        b_px = min(img.height, int(y_max * img.height) + padding)
                    else:
                        # Absolute coordinates in points
                        logger.debug(f"Using absolute coordinates with origin: {coord_origin}")
                        
                        # Get page dimensions to convert points to normalized coordinates
                        # Docling coordinates are in points (1/72 inch)
                        # We need to normalize them based on the page size
                        
                        # For now, assume the page is standard size and coordinates are already in right scale
                        # We need to get actual page dimensions from the page object
                        if hasattr(page, 'size'):
                            page_width = float(page.size.width)
                            page_height = float(page.size.height)
                            logger.debug(f"Page size: {page_width} x {page_height}")
                        else:
                            # Fallback to common PDF page size (US Letter: 612x792 points)
                            page_width = 612.0
                            page_height = 792.0
                            logger.warning(f"Could not get page size, using default: {page_width} x {page_height}")
                        
                        # Normalize coordinates to 0-1 range
                        l_norm = l / page_width
                        r_norm = r / page_width
                        t_norm = t / page_height
                        b_norm = b / page_height
                        
                        # Handle coordinate origin transformation
                        if coord_origin == "BOTTOMLEFT":
                            # Convert from bottom-left origin to top-left origin
                            # In bottom-left: y increases upward
                            # In top-left: y increases downward
                            # So we need to flip: y_topleft = 1 - y_bottomleft
                            t_norm_flipped = 1 - t_norm
                            b_norm_flipped = 1 - b_norm
                            
                            # After flipping, top and bottom are swapped, so swap them back
                            y_min = min(t_norm_flipped, b_norm_flipped)
                            y_max = max(t_norm_flipped, b_norm_flipped)
                        else:
                            # Already in top-left origin
                            y_min = min(t_norm, b_norm)
                            y_max = max(t_norm, b_norm)
                        
                        x_min = min(l_norm, r_norm)
                        x_max = max(l_norm, r_norm)
                        
                        # Scale to image size
                        l_px = max(0, int(x_min * img.width) - padding)
                        r_px = min(img.width, int(x_max * img.width) + padding)
                        t_px = max(0, int(y_min * img.height) - padding)
                        b_px = min(img.height, int(y_max * img.height) + padding)
                    
                elif 'x' in bbox_coords and 'y' in bbox_coords and 'w' in bbox_coords and 'h' in bbox_coords:
                    # Alternative format with x, y, width, height
                    x = float(bbox_coords.get("x", 0))
                    y = float(bbox_coords.get("y", 0))
                    w = float(bbox_coords.get("w", 0))
                    h = float(bbox_coords.get("h", 0))
                    
                    # Check if normalized or absolute
                    if x <= 1 and y <= 1 and w <= 1 and h <= 1:
                        # Normalized
                        l_px = max(0, int(x * img.width) - padding)
                        r_px = min(img.width, int((x + w) * img.width) + padding)
                        t_px = max(0, int(y * img.height) - padding)
                        b_px = min(img.height, int((y + h) * img.height) + padding)
                    else:
                        # Absolute - assume already in right scale
                        l_px = max(0, int(x) - padding)
                        r_px = min(img.width, int(x + w) + padding)
                        t_px = max(0, int(y) - padding)
                        b_px = min(img.height, int(y + h) + padding)
                else:
                    logger.warning(f"Bbox {idx} has unknown format: {bbox_coords}")
                    continue
                
                # Final validation
                if l_px >= r_px or t_px >= b_px:
                    logger.warning(f"Invalid bbox after transformation: ({l_px}, {t_px}) to ({r_px}, {b_px}) - original: {bbox_coords}")
                    continue
                
                logger.debug(f"Drawing box at pixels: ({l_px}, {t_px}) to ({r_px}, {b_px})")
                
                # Draw rectangle with blue outline
                draw.rectangle(
                    xy=[(l_px, t_px), (r_px, b_px)],
                    outline="blue",
                    width=thickness,
                )
                boxes_drawn += 1
            
            logger.info(f"Successfully drew {boxes_drawn}/{len(bboxes)} bounding boxes on page {page_no}")
            return img
            
        except Exception as e:
            logger.error(f"Failed to draw bounding boxes on page {page_no}: {e}", exc_info=True)
            raise
    
    @staticmethod
    def extract_provenance_from_chunk_metadata(chunk_metadata: dict) -> list[dict]:
        """
        Extract provenance information from chunk metadata.
        
        Args:
            chunk_metadata: Metadata dictionary from a document chunk
            
        Returns:
            List of provenance dictionaries with page_no and bbox
        """
        provenances = []
        
        try:
            dl_meta = chunk_metadata.get("dl_meta", {})
            doc_items = dl_meta.get("doc_items", [])
            
            for item in doc_items:
                prov_list = item.get("prov", [])
                for prov in prov_list:
                    if prov and "page_no" in prov:
                        provenances.append(prov)
                        
        except Exception as e:
            logger.warning(f"Could not extract provenance: {e}")
            
        return provenances


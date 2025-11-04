"""
Service layer for document processing, embeddings, and vector storage.
"""

import tempfile
import time
from pathlib import Path
from typing import BinaryIO

import numpy as np
from docling.datamodel.accelerator_options import AcceleratorDevice, AcceleratorOptions
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
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
    ) -> tuple[str, DocumentMetadata]:
        """Process PDF and return markdown with metadata."""
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
            return markdown_content, metadata

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
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.embedding_dim),
            ]
            schema = CollectionSchema(fields, "PDF RAG Collection")

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
        """Ingest documents."""
        if self.collection is None:
            raise VectorStoreError("Collection not initialized")

        try:
            logger.info(f"Ingesting {len(chunks)} chunks")
            texts = [chunk.text for chunk in chunks]
            entities = [texts, embeddings]
            self.collection.insert(entities)
            self.collection.load()
            logger.info(f"Ingested {len(chunks)} chunks")
            return len(chunks)

        except Exception as e:
            logger.error(f"Ingestion failed: {e}")
            raise VectorStoreError(f"Ingestion failed: {e}") from e

    def search(self, query_embedding: list[float], top_k: int = 3) -> list[SearchResult]:
        """Search for similar chunks."""
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
                        expr=f"id == {hit.id}", output_fields=["text"]
                    )
                    if result_entity:
                        search_results.append(
                            SearchResult(
                                text=result_entity[0]["text"],
                                score=float(hit.score),
                                chunk_id=int(hit.id),
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

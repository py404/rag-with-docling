"""
Pydantic models for configuration and data validation.
"""

from enum import Enum
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, field_validator
from pydantic_settings import BaseSettings


class AcceleratorType(str, Enum):
    """Supported accelerator types."""

    MPS = "mps"
    CUDA = "cuda"
    CPU = "cpu"


class ProcessingConfig(BaseModel):
    """PDF processing configuration."""

    use_ocr: bool = Field(default=True, description="Enable OCR")
    use_table_structure: bool = Field(default=True, description="Extract tables")
    num_threads: int = Field(default=8, ge=1, le=32, description="Thread count")
    accelerator: AcceleratorType = Field(default=AcceleratorType.MPS)

    @field_validator("num_threads")
    @classmethod
    def validate_threads(cls, v: int) -> int:
        """Validate thread count."""
        if not 1 <= v <= 32:
            raise ValueError("Thread count must be between 1 and 32")
        return v


class EmbeddingConfig(BaseModel):
    """Embedding configuration."""

    model_name: str = Field(default="all-MiniLM-L6-v2")
    dimension: int = Field(default=384, ge=128, le=1536)
    normalize: bool = Field(default=True)
    device: str = Field(default="mps")


class VectorStoreConfig(BaseModel):
    """Vector database configuration."""

    uri: str = Field(default="rag.db")
    collection_name: str = Field(default="docling_rag")
    index_type: str = Field(default="IVF_FLAT")
    metric_type: str = Field(default="L2")
    nlist: int = Field(default=1024, ge=128)
    nprobe: int = Field(default=10, ge=1)

    @field_validator("collection_name")
    @classmethod
    def validate_collection_name(cls, v: str) -> str:
        """Validate collection name."""
        if not v or not v.replace("_", "").isalnum():
            raise ValueError("Collection name must be alphanumeric with underscores")
        return v


class SearchConfig(BaseModel):
    """Search configuration."""

    top_k: int = Field(default=3, ge=1, le=20)
    min_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)


class AppSettings(BaseSettings):
    """Application settings."""

    app_title: str = Field(default="PDF RAG with Docling")
    app_icon: str = Field(default="📚")
    max_file_size_mb: int = Field(default=50, ge=1)
    temp_dir: Path = Field(default=Path("/tmp"))

    processing: ProcessingConfig = Field(default_factory=ProcessingConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    vector_store: VectorStoreConfig = Field(default_factory=VectorStoreConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)

    class Config:
        """Pydantic config."""

        env_prefix = "APP_"
        case_sensitive = False


class DocumentMetadata(BaseModel):
    """Document metadata."""

    filename: str
    source: str
    char_count: int = Field(ge=0)
    chunk_count: int = Field(ge=0)
    has_tables: bool = Field(default=False)
    table_count: int = Field(default=0, ge=0)
    processing_time: float = Field(ge=0.0)
    mean_quality: str
    low_quality: str


class SearchResult(BaseModel):
    """Search result."""

    text: str
    score: float = Field(ge=0.0)
    chunk_id: int = Field(ge=0)
    metadata: dict = Field(default_factory=dict, description="Chunk metadata with provenance")

    @property
    def preview(self) -> str:
        """Get text preview."""
        return self.text[:200] + "..." if len(self.text) > 200 else self.text


class BoundingBox(BaseModel):
    """Bounding box coordinates."""

    l: float = Field(description="Left coordinate")
    t: float = Field(description="Top coordinate")
    r: float = Field(description="Right coordinate")
    b: float = Field(description="Bottom coordinate")


class ProvenanceInfo(BaseModel):
    """Provenance information for document items."""

    page_no: int = Field(description="Page number")
    bbox: Optional[dict] = Field(default=None, description="Bounding box coordinates")


class DocItemInfo(BaseModel):
    """Document item information."""

    prov: Optional[list[ProvenanceInfo]] = Field(default=None, description="Provenance list")


class DocMeta(BaseModel):
    """Document metadata for visual grounding."""

    origin: dict = Field(description="Document origin information")
    doc_items: list[DocItemInfo] = Field(default_factory=list, description="Document items")

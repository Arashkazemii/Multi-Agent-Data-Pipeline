# ingestor.py
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Iterable, List, Optional, Dict, Any, Tuple

from dotenv import load_dotenv
from haystack import Document
from haystack.utils import Secret

# Storage & components (no retrievers here)
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.preprocessors import DocumentPreprocessor
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter

# Optional basic converters (extend as needed)
from haystack.components.converters import TextFileToDocument

load_dotenv()
OPENAI_API_KEY = Secret.from_env_var("OPENAI_API_KEY")

# ============ Configs ============ 

@dataclass(frozen=True)
class ChunkingConfig:
    # Token-oriented: ~220 words ≈ ~350 tokens on average
    split_by: str = "word"
    split_length: int = 220
    split_overlap: int = 40
    language: str = "en"
    splitting_function: Optional[str] = None

@dataclass(frozen=True)
class EmbeddingConfig:
    model: str = "text-embedding-3-small"
    dimensions: int = 1536
    batch_size: int = 64
    embedding_separator: str = "\n"

@dataclass(frozen=True)
class StoreConfig:
    # InMemory only here; swap later for FAISS/Weaviate/etc.
    return_embedding: bool = True  # keep embeddings accessible on reads


# ============ Builders ============

def load_document_store(cfg: StoreConfig = StoreConfig()) -> InMemoryDocumentStore:
    """
    In-memory store for development. Replace with your persistent store in prod.
    """
    return InMemoryDocumentStore(return_embedding=cfg.return_embedding)

def load_preprocessor(cfg: ChunkingConfig = ChunkingConfig()) -> DocumentPreprocessor:
    """
    Sentence/word chunker tuned for RAG-friendly chunk sizes.
    """
    return DocumentPreprocessor(
        split_by=cfg.split_by,
        split_length=cfg.split_length,
        split_overlap=cfg.split_overlap,
        language=cfg.language,
        splitting_function=cfg.splitting_function,
        progress_bar=True,
    )

def load_doc_embedder(
    cfg: EmbeddingConfig = EmbeddingConfig(),
    api_key: Secret = OPENAI_API_KEY,
) -> OpenAIDocumentEmbedder:
    """
    Document embedder (for chunks). Query embedder is part of the retriever module.
    """
    return OpenAIDocumentEmbedder(
        api_key=api_key,
        model=cfg.model,
        dimensions=cfg.dimensions,
        batch_size=cfg.batch_size,
        embedding_separator=cfg.embedding_separator,
    )


# ============ Utilities ============

def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

def _dedupe_by_content(docs: List[Document]) -> Tuple[List[Document], int]:
    """
    Deduplicate exact duplicate contents before chunking/embedding.
    """
    seen: set[str] = set()
    unique_docs: List[Document] = []
    dup_count = 0
    for d in docs:
        h = _content_hash(d.content or "")
        if h in seen:
            dup_count += 1
            continue
        seen.add(h)
        unique_docs.append(d)
    return unique_docs, dup_count


# ============ Public Ingestion API ============

def ingest_texts(
    texts: Iterable[str],
    *,
    store: InMemoryDocumentStore,
    preprocessor: DocumentPreprocessor,
    doc_embedder: OpenAIDocumentEmbedder,
    default_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Ingest raw strings. Returns an ingestion report dict.
    """
    base_docs = [Document(content=t, meta=dict(default_metadata or {})) for t in texts]
    unique_docs, dup_count = _dedupe_by_content(base_docs)

    # Preprocess -> Embed -> Write
    pre_out = preprocessor.run(documents=unique_docs)
    chunks: List[Document] = pre_out["documents"]

    emb_out = doc_embedder.run(documents=chunks)
    embedded_chunks: List[Document] = emb_out["documents"]

    DocumentWriter(document_store=store).run(documents=embedded_chunks)

    return {
        "input_docs": len(base_docs),
        "duplicates_skipped": dup_count,
        "chunks_written": len(embedded_chunks),
        "avg_chunk_chars": int(sum(len(d.content or "") for d in embedded_chunks) / max(1, len(embedded_chunks))),
        "embedding_model": doc_embedder.model,  # type: ignore[attr-defined]
        "store_type": type(store).__name__,
    }

def ingest_text_files(
    file_paths: Iterable[str],
    *,
    store: InMemoryDocumentStore,
    preprocessor: DocumentPreprocessor,
    doc_embedder: OpenAIDocumentEmbedder,
    default_metadata: Optional[Dict[str, Any]] = None,
    encoding: str = "utf-8",
) -> Dict[str, Any]:
    """
    Ingest plain text files. Extend with PyPDFToDocument/HTMLToDocument as needed.
    """
    converter = TextFileToDocument(encoding=encoding)
    conv_out = converter.run(paths=list(file_paths))
    docs: List[Document] = conv_out["documents"]

    # Attach default metadata without clobbering file-provided meta
    for d in docs:
        d.meta = {**(default_metadata or {}), **(d.meta or {})}

    unique_docs, dup_count = _dedupe_by_content(docs)

    pre_out = preprocessor.run(documents=unique_docs)
    chunks: List[Document] = pre_out["documents"]

    emb_out = doc_embedder.run(documents=chunks)
    embedded_chunks: List[Document] = emb_out["documents"]

    DocumentWriter(document_store=store).run(documents=embedded_chunks)

    return {
        "files_read": len(list(file_paths)),
        "source_docs": len(docs),
        "duplicates_skipped": dup_count,
        "chunks_written": len(embedded_chunks),
        "avg_chunk_chars": int(sum(len(d.content or "") for d in embedded_chunks) / max(1, len(embedded_chunks))),
        "embedding_model": doc_embedder.model,  # type: ignore[attr-defined]
        "store_type": type(store).__name__,
    }

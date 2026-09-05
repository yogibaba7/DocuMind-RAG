"""
RAG Pipeline Scripts Package
Exports retriever, generation, and conversational RAG chain components.
"""

from .retriever import (
    build_retriever,
    create_vector_store,
    create_embedding_model,
    HybridHyDeRetriever,
    HyDeModel,
    clean_text,
    save_metrics,
)

from .generation import (
    create_model,
    get_available_models,
    AVAILABLE_MODELS,
)

from .rag_chain import (
    create_rag_chain,
    create_contextualize_chain,
    create_answer_chain,
    format_docs_with_citations,
    format_docs_simple,
    ConversationalRAGEngine,
)

__all__ = [
    "build_retriever",
    "create_vector_store",
    "create_embedding_model",
    "HybridHyDeRetriever",
    "HyDeModel",
    "clean_text",
    "save_metrics",
    "create_model",
    "get_available_models",
    "AVAILABLE_MODELS",
    "create_rag_chain",
    "create_contextualize_chain",
    "create_answer_chain",
    "format_docs_with_citations",
    "format_docs_simple",
    "ConversationalRAGEngine",
]

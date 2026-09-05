import os
import re
import time
import csv
import hashlib
import unicodedata
from datetime import datetime
from typing import List, Optional, Tuple, Any, Dict

from dotenv import load_dotenv
from pydantic import Field

from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from sentence_transformers import CrossEncoder

load_dotenv()

# ============================================================
# CONFIGURATION CONSTANTS
# ============================================================

METRICS_FILE = "indexing_metrics.csv"
BASE_FAISS_DIR = "faiss_index"
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_HYDE_MODEL = "qwen/qwen3.8-27b"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 5
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
RERANK_TOP_K_MULTIPLIER = 3  # Retrieve TOP_K * multiplier candidates, then rerank to TOP_K


# ============================================================
# UTILITIES: TEXT CLEANING & BENCHMARK METRICS
# ============================================================

def clean_text(text: str) -> str:
    """
    Cleans noisy whitespace, unicode artifacts, and hyphenation from extracted PDF text.
    """
    if not text:
        return ""
    text = unicodedata.normalize("NFKD", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u2010", "-").replace("\u2013", "-").replace("\u2014", "-")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def get_file_hash(file_path: str) -> str:
    """Computes SHA256 hash prefix for document versioning and caching."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        while chunk := f.read(8192):
            hasher.update(chunk)
    return hasher.hexdigest()[:12]


def save_metrics(metrics: Dict[str, Any]) -> None:
    """Saves indexing benchmark metrics to CSV for performance tracking."""
    file_exists = os.path.exists(METRICS_FILE)
    fieldnames = list(metrics.keys())

    if not file_exists:
        with open(METRICS_FILE, "w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerow(metrics)
        return

    with open(METRICS_FILE, "r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        old_fieldnames = reader.fieldnames or []
        old_records = list(reader)

    for field in fieldnames:
        if field not in old_fieldnames:
            old_fieldnames.append(field)

    with open(METRICS_FILE, "w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=old_fieldnames)
        writer.writeheader()
        for record in old_records:
            writer.writerow(record)
        writer.writerow(metrics)


# ============================================================
# HYDE (Hypothetical Document Embeddings)
# ============================================================

def HyDeModel(query: str, model_name: str = DEFAULT_HYDE_MODEL) -> str:
    """
    Generates a hypothetical document passage for the query to enhance
    semantic dense vector retrieval.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        print("Warning: GROQ_API_KEY not found. Falling back to raw query for retrieval.")
        return query

    prompt = ChatPromptTemplate.from_template(
        """Write a concise hypothetical answer or relevant document passage that directly answers the user's question.
This passage will be used for dense semantic retrieval. Focus on factual phrasing and vocabulary that would appear in a reliable document.

Question:
{query}

Hypothetical passage:"""
    )

    prompted_query = prompt.invoke({"query": query})

    llm = ChatGroq(
        model=model_name,
        temperature=0,
        max_tokens=384,
        groq_api_key=groq_api_key,
    )

    try:
        response = llm.invoke(prompted_query).content
        cleaned = response.strip()
        return cleaned if cleaned else query
    except Exception as e:
        print(f"HyDE expansion warning ({e}), falling back to raw query.")
        return query


# ============================================================
# EMBEDDINGS & VECTOR STORE
# ============================================================

def create_embedding_model(model_name: str = DEFAULT_EMBEDDING_MODEL) -> HuggingFaceEndpointEmbeddings:
    """Initializes HuggingFace endpoint embeddings model."""
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError(
            "HUGGINGFACE_API_KEY not found in .env file. "
            "Please set HUGGINGFACE_API_KEY to generate embeddings."
        )

    return HuggingFaceEndpointEmbeddings(
        model=model_name,
        huggingfacehub_api_token=hf_token,
    )


def create_vector_store(
    document_path: str = "data/sample_policies.pdf",
    force_rebuild: bool = False,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    embedding_model_name: str = DEFAULT_EMBEDDING_MODEL,
) -> Tuple[FAISS, BM25Retriever, Dict[str, Any]]:
    """
    Loads, cleans, and chunks the PDF document, building both:
    1. A dense FAISS vector store (cached per document hash).
    2. A sparse BM25 keyword retriever.
    3. Indexing performance metrics dictionary.
    """
    if not os.path.exists(document_path):
        raise FileNotFoundError(f"Document not found at path: {document_path}")

    print(f"\n[Indexing] Processing Document: {document_path} ...")
    total_start = time.perf_counter()

    document_name = os.path.basename(document_path)
    document_size_bytes = os.path.getsize(document_path)
    document_size_mb = document_size_bytes / (1024 * 1024)
    doc_hash = get_file_hash(document_path)

    # Document-specific index directory to avoid collisions across different files
    safe_name = re.sub(r"[^\w\-_\.]", "_", os.path.splitext(document_name)[0])
    doc_index_dir = os.path.join(BASE_FAISS_DIR, f"{safe_name}_{doc_hash}")

    # 1. Load document
    load_start = time.perf_counter()
    loader = PyMuPDFLoader(document_path)
    documents = loader.load()

    # Clean text and standardize metadata (e.g. 1-indexed page number for UI display)
    for doc in documents:
        doc.page_content = clean_text(doc.page_content)
        # Ensure page number is human friendly (1-based)
        if "page" in doc.metadata and isinstance(doc.metadata["page"], int):
            doc.metadata["page_number"] = doc.metadata["page"] + 1
        else:
            doc.metadata["page_number"] = 1
        doc.metadata["source_file"] = document_name

    loading_time = time.perf_counter() - load_start
    loading_mb_per_sec = document_size_mb / loading_time if loading_time > 0 else 0

    # 2. Chunk document
    chunk_start = time.perf_counter()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    chunking_time = time.perf_counter() - chunk_start
    chunking_mb_per_sec = document_size_mb / chunking_time if chunking_time > 0 else 0

    # 3. Create BM25 Keyword Retriever
    keyword_retriever = BM25Retriever.from_documents(chunks)
    keyword_retriever.k = TOP_K

    # 4. Dense Vector Store (FAISS)
    embeddings = create_embedding_model(model_name=embedding_model_name)
    index_exists = (
        os.path.exists(doc_index_dir)
        and os.path.exists(os.path.join(doc_index_dir, "index.faiss"))
        and os.path.exists(os.path.join(doc_index_dir, "index.pkl"))
    )

    if index_exists and not force_rebuild:
        print(f"[Indexing] Loading cached FAISS index from '{doc_index_dir}'...")
        start_faiss = time.perf_counter()
        vector_store = FAISS.load_local(
            doc_index_dir,
            embeddings,
            allow_dangerous_deserialization=True,
        )
        embedding_time = time.perf_counter() - start_faiss
        print(f"[Indexing] FAISS cache loaded in {embedding_time:.2f}s.")
    else:
        print(f"[Indexing] Generating embeddings and building FAISS index at '{doc_index_dir}'...")
        start_faiss = time.perf_counter()
        vector_store = FAISS.from_documents(chunks, embeddings)
        os.makedirs(doc_index_dir, exist_ok=True)
        vector_store.save_local(doc_index_dir)
        embedding_time = time.perf_counter() - start_faiss
        print(f"[Indexing] FAISS index saved in {embedding_time:.2f}s.")

    embedding_mb_per_sec = document_size_mb / embedding_time if embedding_time > 0 else 0
    total_time = time.perf_counter() - total_start
    total_mb_per_sec = document_size_mb / total_time if total_time > 0 else 0

    # 5. Record & return metrics
    metrics = {
        "timestamp": datetime.now().isoformat(),
        "document_name": document_name,
        "document_hash": doc_hash,
        "document_size_mb": round(document_size_mb, 4),
        "pages": len(documents),
        "chunks": len(chunks),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": embedding_model_name,
        "loading_time_sec": round(loading_time, 4),
        "loading_mb_per_sec": round(loading_mb_per_sec, 4),
        "chunking_time_sec": round(chunking_time, 4),
        "chunking_mb_per_sec": round(chunking_mb_per_sec, 4),
        "embedding_time_sec": round(embedding_time, 4),
        "embedding_mb_per_sec": round(embedding_mb_per_sec, 4),
        "total_time_sec": round(total_time, 4),
        "total_mb_per_sec": round(total_mb_per_sec, 4),
    }
    save_metrics(metrics)

    print("\n========== INDEXING BENCHMARK ==========")
    print(f"Document       : {document_name} ({document_size_mb:.2f} MB)")
    print(f"Pages / Chunks : {len(documents)} pages / {len(chunks)} chunks")
    print(f"Total Time     : {total_time:.2f}s ({total_mb_per_sec:.2f} MB/s)")
    print("========================================\n")

    return vector_store, keyword_retriever, metrics


# ============================================================
# HYBRID + HYDE RETRIEVER CLASS
# ============================================================

class HybridHyDeRetriever(BaseRetriever):
    """
    Advanced Hybrid Retriever with Reranking:
    1. Query -> HyDE: Generates hypothetical document passage.
    2. HyDE / Query -> Dense FAISS Vector Search + Sparse BM25 Keyword Search (oversampled).
    3. Merges and deduplicates retrieved chunks.
    4. Reranks with a CrossEncoder to push the most relevant chunks to the top.
    5. Returns the top_k best chunks.
    """
    vector_retriever: Any = Field(description="FAISS vector retriever")
    keyword_retriever: Any = Field(description="BM25 keyword retriever")
    reranker: Any = Field(default=None, description="CrossEncoder reranker model")
    use_hyde: bool = Field(default=True, description="Whether to apply HyDE expansion")
    hyde_model: str = Field(default=DEFAULT_HYDE_MODEL, description="LLM model used for HyDE")
    top_k: int = Field(default=TOP_K, description="Number of unique documents to return")
    oversample_k: int = Field(default=TOP_K * RERANK_TOP_K_MULTIPLIER, description="Number of candidates to retrieve before reranking")

    def _get_relevant_documents(
        self, query: str, *, run_manager: Optional[CallbackManagerForRetrieverRun] = None
    ) -> List[Document]:
        # Step 1: HyDE reformulation
        if self.use_hyde:
            search_query = HyDeModel(query, model_name=self.hyde_model)
            print(f"    [HyDE] Expanded query: {search_query[:80]}...")
        else:
            search_query = query

        # Step 2: Dense + Sparse Retrieval (oversampled for reranking)
        vector_results = self.vector_retriever.invoke(search_query)
        keyword_results = self.keyword_retriever.invoke(query)

        # Step 3: Combine & Deduplicate by page content
        combined_results = vector_results + keyword_results
        unique_results: List[Document] = []
        seen_contents = set()

        for doc in combined_results:
            normalized = clean_text(doc.page_content)
            if normalized and normalized not in seen_contents:
                seen_contents.add(normalized)
                unique_results.append(doc)

        # Step 4: Rerank with CrossEncoder (if available)
        if self.reranker is not None and len(unique_results) > 1:
            # Create query-document pairs for the cross-encoder
            pairs = [(query, doc.page_content) for doc in unique_results]
            scores = self.reranker.predict(pairs)

            # Sort by reranker score (highest first)
            scored_docs = sorted(
                zip(scores, unique_results),
                key=lambda x: x[0],
                reverse=True,
            )

            reranked = [doc for _, doc in scored_docs]
            return reranked[: self.top_k]

        # Fallback: no reranker, return as-is
        return unique_results[: self.top_k]


def build_retriever(
    document_path: str = "data/sample_policies.pdf",
    force_rebuild: bool = False,
    top_k: int = TOP_K,
    use_hyde: bool = True,
    hyde_model: str = DEFAULT_HYDE_MODEL,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    rerank_model: str = RERANK_MODEL,
    use_reranker: bool = True,
) -> Tuple[HybridHyDeRetriever, Dict[str, Any]]:
    """
    Constructs and returns the HybridHyDeRetriever and indexing performance metrics.
    Pipeline: HyDE -> (FAISS + BM25 oversampled) -> Deduplicate -> Rerank -> top_k
    """
    vector_store, keyword_retriever, metrics = create_vector_store(
        document_path=document_path,
        force_rebuild=force_rebuild,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Oversample: retrieve more candidates for the reranker to pick from
    oversample_k = top_k * RERANK_TOP_K_MULTIPLIER if use_reranker else top_k

    vector_retriever = vector_store.as_retriever(search_kwargs={"k": oversample_k})
    keyword_retriever.k = oversample_k

    # Load cross-encoder reranker
    reranker = None
    if use_reranker:
        print(f"[Reranker] Loading CrossEncoder: {rerank_model}...")
        reranker = CrossEncoder(rerank_model)
        print(f"[Reranker] Ready. Will rerank {oversample_k} candidates -> top {top_k}.")

    retriever = HybridHyDeRetriever(
        vector_retriever=vector_retriever,
        keyword_retriever=keyword_retriever,
        reranker=reranker,
        use_hyde=use_hyde,
        hyde_model=hyde_model,
        top_k=top_k,
        oversample_k=oversample_k,
    )

    return retriever, metrics


if __name__ == "__main__":
    test_pdf = "data/sample_policies.pdf"
    if os.path.exists(test_pdf):
        print(f"Testing retriever with {test_pdf}...")
        retriever, metrics = build_retriever(document_path=test_pdf, top_k=3, use_hyde=False)
        sample_results = retriever.invoke("leave policy")
        print(f"Retrieved {len(sample_results)} results:")
        for idx, doc in enumerate(sample_results, 1):
            print(f"[{idx}] Page {doc.metadata.get('page_number', 'N/A')}: {doc.page_content[:150]}...\n")
    else:
        print(f"File '{test_pdf}' not found for direct test.")
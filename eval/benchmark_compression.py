"""
Benchmark Script: Contextual Compression vs Baseline RAG Evaluation
Measures the system-level impact of adding an LLM-based verbatim context extractor:
- Context Length (Characters & Est. Tokens)
- Noise Reduction & Chunk Pruning Rate
- Component & End-to-End Latencies
- Answer Groundedness & Conciseness
"""
import os
import sys
import time
import json
from typing import Dict, Any, List

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.retriever import build_retriever
from scripts.generation import create_model
from scripts.rag_chain import (
    format_docs_with_citations,
    compress_documents,
    ANSWER_SYSTEM_PROMPT,
)
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

TEST_QUESTIONS = [
    "What is the company's policy regarding sexual harassment and what disciplinary actions can result from a violation?",
    "What are the conditions under which an employee can resign?",
    "What is the company's policy on taking leave, including eligibility and approval requirements?",
    "What are the consequences of violating the company's code of conduct?",
    "What is the company's policy regarding conflicts of interest?",
    "What are the employee's rights and responsibilities regarding confidential information?",
    "Under what circumstances can an employee's employment be terminated?",
]


def run_benchmark():
    print("=" * 80)
    print("DOCUMIND-RAG: CONTEXTUAL COMPRESSION SYSTEM IMPACT BENCHMARK")
    print("=" * 80)

    pdf_path = os.path.join(ROOT_DIR, "data", "sample_policies.pdf")
    print(f"Loading retriever with benchmark document: {pdf_path}")
    retriever, metrics = build_retriever(document_path=pdf_path, top_k=5, use_hyde=True)
    
    print("Initializing LLM model (qwen/qwen3.8-27b)...")
    model = create_model()

    answer_prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    answer_chain = answer_prompt | model

    results: List[Dict[str, Any]] = []

    for idx, query in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "-" * 80)
        print(f"[{idx}/{len(TEST_QUESTIONS)}] Question: {query}")
        print("-" * 80)

        # 1. Retrieval Phase
        t0 = time.time()
        retrieved_docs = retriever.invoke(query)
        retrieval_time = time.time() - t0
        base_chunk_count = len(retrieved_docs)
        base_context_chars = sum(len(d.page_content) for d in retrieved_docs)
        base_context_text = format_docs_with_citations(retrieved_docs)
        print(f"Retrieved {base_chunk_count} chunks ({base_context_chars} chars) in {retrieval_time:.2f}s")

        # 2. Baseline Generation (Without Compression)
        t1 = time.time()
        base_resp = answer_chain.invoke({
            "context": base_context_text,
            "chat_history": [],
            "input": query,
        })
        base_gen_time = time.time() - t1
        base_answer = base_resp.content if hasattr(base_resp, "content") else str(base_resp)
        base_total_time = retrieval_time + base_gen_time
        print(f"Baseline Answer generated in {base_gen_time:.2f}s (Total: {base_total_time:.2f}s, {len(base_answer)} chars)")

        time.sleep(1.0)

        # 3. Contextual Compression Phase
        t2 = time.time()
        compressed_docs = compress_documents(retrieved_docs, query, model)
        compression_time = time.time() - t2
        comp_chunk_count = len(compressed_docs)
        comp_context_chars = sum(len(d.page_content) for d in compressed_docs)
        comp_context_text = format_docs_with_citations(compressed_docs)
        
        pruned_chunks = base_chunk_count - comp_chunk_count
        char_reduction_pct = (
            ((base_context_chars - comp_context_chars) / base_context_chars * 100)
            if base_context_chars > 0 else 0.0
        )
        print(f"Compression finished in {compression_time:.2f}s:")
        print(f"  Retained: {comp_chunk_count}/{base_chunk_count} chunks (Pruned: {pruned_chunks})")
        print(f"  Context chars: {base_context_chars} -> {comp_context_chars} ({char_reduction_pct:.1f}% reduction)")

        time.sleep(1.0)

        # 4. Compressed Generation (With Compression)
        t3 = time.time()
        comp_resp = answer_chain.invoke({
            "context": comp_context_text,
            "chat_history": [],
            "input": query,
        })
        comp_gen_time = time.time() - t3
        comp_answer = comp_resp.content if hasattr(comp_resp, "content") else str(comp_resp)
        comp_total_time = retrieval_time + compression_time + comp_gen_time
        print(f"Compressed Answer generated in {comp_gen_time:.2f}s (Total: {comp_total_time:.2f}s, {len(comp_answer)} chars)")

        # Collect record
        record = {
            "query_id": idx,
            "question": query,
            "retrieval_time_sec": round(retrieval_time, 3),
            "baseline": {
                "chunk_count": base_chunk_count,
                "context_char_length": base_context_chars,
                "est_context_tokens": round(base_context_chars / 4),
                "generation_time_sec": round(base_gen_time, 3),
                "total_latency_sec": round(base_total_time, 3),
                "answer_char_length": len(base_answer),
                "answer": base_answer,
                "chunks": [{"page": d.metadata.get("page_number"), "text": d.page_content} for d in retrieved_docs],
            },
            "compressed": {
                "compression_time_sec": round(compression_time, 3),
                "chunk_count": comp_chunk_count,
                "pruned_chunks_count": pruned_chunks,
                "context_char_length": comp_context_chars,
                "est_context_tokens": round(comp_context_chars / 4),
                "char_reduction_pct": round(char_reduction_pct, 2),
                "generation_time_sec": round(comp_gen_time, 3),
                "total_latency_sec": round(comp_total_time, 3),
                "answer_char_length": len(comp_answer),
                "answer": comp_answer,
                "chunks": [{"page": d.metadata.get("page_number"), "text": d.page_content} for d in compressed_docs],
            },
        }
        results.append(record)

    # Summary Statistics
    avg_base_context = sum(r["baseline"]["context_char_length"] for r in results) / len(results)
    avg_comp_context = sum(r["compressed"]["context_char_length"] for r in results) / len(results)
    overall_reduction = ((avg_base_context - avg_comp_context) / avg_base_context) * 100
    avg_pruned = sum(r["compressed"]["pruned_chunks_count"] for r in results) / len(results)
    avg_comp_time = sum(r["compressed"]["compression_time_sec"] for r in results) / len(results)
    avg_base_gen = sum(r["baseline"]["generation_time_sec"] for r in results) / len(results)
    avg_comp_gen = sum(r["compressed"]["generation_time_sec"] for r in results) / len(results)
    avg_base_total = sum(r["baseline"]["total_latency_sec"] for r in results) / len(results)
    avg_comp_total = sum(r["compressed"]["total_latency_sec"] for r in results) / len(results)

    summary = {
        "total_queries": len(results),
        "avg_baseline_context_chars": round(avg_base_context, 1),
        "avg_compressed_context_chars": round(avg_comp_context, 1),
        "avg_context_reduction_pct": round(overall_reduction, 2),
        "avg_chunks_pruned_per_query": round(avg_pruned, 2),
        "avg_compression_overhead_sec": round(avg_comp_time, 2),
        "avg_baseline_generation_time_sec": round(avg_base_gen, 2),
        "avg_compressed_generation_time_sec": round(avg_comp_gen, 2),
        "avg_baseline_total_latency_sec": round(avg_base_total, 2),
        "avg_compressed_total_latency_sec": round(avg_comp_total, 2),
    }

    output_path = os.path.join(ROOT_DIR, "eval", "compression_benchmark_results.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("BENCHMARK SUMMARY RESULTS")
    print("=" * 80)
    print(f"Average Baseline Context:   {avg_base_context:.1f} chars (~{int(avg_base_context/4)} tokens)")
    print(f"Average Compressed Context: {avg_comp_context:.1f} chars (~{int(avg_comp_context/4)} tokens)")
    print(f"Average Context Reduction:  {overall_reduction:.2f}%")
    print(f"Average Chunks Pruned:      {avg_pruned:.1f} out of 5 chunks")
    print(f"Avg Compression Overhead:   {avg_comp_time:.2f}s")
    print(f"Avg Generator LLM Latency:  {avg_base_gen:.2f}s (baseline) -> {avg_comp_gen:.2f}s (compressed)")
    print(f"Avg End-to-End Latency:     {avg_base_total:.2f}s (baseline) -> {avg_comp_total:.2f}s (compressed)")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    run_benchmark()
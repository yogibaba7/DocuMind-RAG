"""
Benchmark Script: Evaluating Prompt Optimization (V1 vs V2) With Contextual Compression
Measures the system-level impact of optimizing:
  1. Context Compressor Prompt (V1 vs V2)
  2. Generator QA Prompt (V1 vs V2)
Evaluated across 7 benchmark policy questions using Groq qwen/qwen3.8-27b.
"""
import os
import sys
import time
import json
from typing import Dict, Any, List, Optional

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from scripts.retriever import build_retriever
from scripts.generation import create_model
from scripts.rag_chain import format_docs_with_citations
from eval.graq_judge import GroqJudge
from deepeval.test_case import LLMTestCase
from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric

# ============================================================
# PROMPT DEFINITIONS: V1 (CURRENT) VS V2 (OPTIMIZED)
# ============================================================

COMPRESSOR_PROMPT_V1 = """You are a context extraction system used in a retrieval pipeline.

Your ONLY job is to extract verbatim spans or tightly-scoped excerpts from the DOCUMENT that are relevant to answering the QUESTION.

Rules:
1. Extract only text that is explicitly present in the document — do not paraphrase, summarize, infer, or add anything.
2. Do not answer the question. Do not draw conclusions. Do not explain your reasoning.
3. Preserve the original wording of extracted text exactly as it appears.
4. You may extract multiple non-contiguous snippets if needed; separate them with "..." on their own line.
5. If nothing in the document is relevant, output exactly: NO_RELEVANT_CONTEXT
6. Never output any text other than the extracted content or the NO_RELEVANT_CONTEXT flag — no preamble, no labels, no commentary.

Question:
{query}

Document:
{document}

Return only the relevant information."""

ANSWER_PROMPT_V1 = """You are a highly capable and precise document question-answering assistant.

Your task is to answer the user's question accurately and concisely using ONLY the provided document context below.

Guidelines:
1. Ground your response strictly in the provided context. Do NOT speculate or make up information.
2. If the answer cannot be found or deduced from the provided context, respond politely with:
   "I could not find the answer in the provided document."
3. When helpful, cite the relevant page numbers from the context (e.g., [Page X]).
4. Maintain a clear, professional, and well-structured formatting (use markdown, bullet points, or tables when appropriate).

Context:
{context}"""


COMPRESSOR_PROMPT_V2 = """You are an advanced, high-precision context extraction engine for a retrieval-augmented generation (RAG) system.

Your MISSION is to extract ONLY the exact verbatim sentences or clauses from the DOCUMENT that directly provide facts, rules, conditions, numbers, or answers to the QUESTION.

Strict Extraction Rules:
1. VERBATIM ONLY: Extract exact text as it appears. Never paraphrase, summarize, interpret, or add new words.
2. AGGRESSIVE NOISE FILTERING: Strip out document titles, Roman numeral headers, section numbering, page numbers, bullet numbers, form fields, and signature lines. Extract ONLY the informative factual text.
3. IRRELEVANT CHUNKS: If the document mentions keywords from the question but does NOT provide actual substantive facts answering it, or if nothing is relevant, output EXACTLY:
NO_RELEVANT_CONTEXT
4. MULTI-SPAN EXTRACTION: If multiple non-adjacent sentences are relevant, output each excerpt separated by a single line with "...".
5. DO NOT ANSWER: Do not synthesize, explain, or answer the question yourself.
6. ZERO PREAMBLE: Output ONLY the extracted text or NO_RELEVANT_CONTEXT. No introductory remarks, labels, or notes.

Question:
{query}

Document:
{document}

Extracted Relevant Text:"""

ANSWER_PROMPT_V2 = """You are an executive document question-answering system providing concise, authoritative, and citation-backed answers.

You must answer the user's question using EXCLUSIVELY the provided context excerpts.

Execution Directives:
1. DIRECT ANSWER FIRST (BLUF): State the direct, unequivocal answer in the very first sentence. Never start with conversational filler (e.g., do NOT say "Based on the provided document context", "According to the document", or "Here is what I found").
2. GROUNDING & FIDELITY: Rely strictly on facts explicitly stated in the context. Never speculate, assume, extrapolate, or introduce outside knowledge. If the context does not contain the answer, respond with: "The provided document does not contain this information."
3. PRECISE CITATIONS: Every key fact, rule, number, or condition MUST include its source page citation in square brackets immediately after the statement (e.g., "...within 10 business days [Page 20]."). Never cite pages not explicitly given in the context headers.
4. STRUCTURED SYNTHESIS: Organize details using clean markdown bullet points grouped under bold topical categories (e.g., **Policy / Rule**, **Eligibility**, **Approval Process**, **Disciplinary Actions / Consequences**). Keep sentences crisp and information-dense.

Context:
{context}"""


TEST_QUESTIONS = [
    "What is the company's policy regarding sexual harassment and what disciplinary actions can result from a violation?",
    "What are the conditions under which an employee can resign?",
    "What is the company's policy on taking leave, including eligibility and approval requirements?",
    "What are the consequences of violating the company's code of conduct?",
    "What is the company's policy regarding conflicts of interest?",
    "What are the employee's rights and responsibilities regarding confidential information?",
    "Under what circumstances can an employee's employment be terminated?",
]


def compress_chunk(doc: Document, query: str, prompt_template: str, model) -> Optional[Document]:
    chain = ChatPromptTemplate.from_template(prompt_template) | model | StrOutputParser()
    for attempt in range(3):
        try:
            extracted = chain.invoke({"query": query, "document": doc.page_content}).strip()
            break
        except Exception as e:
            if "429" in str(e) or "rate" in str(e).lower() or attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            return doc
    if not extracted or "NO_RELEVANT_CONTEXT" in extracted.upper():
        return None
    comp_doc = Document(page_content=extracted, metadata=dict(doc.metadata))
    comp_doc.metadata["compressed"] = True
    return comp_doc


def compress_all(docs: List[Document], query: str, prompt_template: str, model) -> List[Document]:
    out = []
    for d in docs:
        c = compress_chunk(d, query, prompt_template, model)
        if c is not None:
            out.append(c)
    return out if out else docs


def run_prompt_experiment():
    print("=" * 80)
    print("DOCUMIND-RAG: PROMPT OPTIMIZATION BENCHMARK (V1 vs V2 - COMPRESSED ONLY)")
    print("=" * 80)

    pdf_path = os.path.join(ROOT_DIR, "data", "sample_policies.pdf")
    print(f"Loading retriever with benchmark document: {pdf_path}")
    retriever, metrics = build_retriever(document_path=pdf_path, top_k=5, use_hyde=True)

    print("Initializing LLM models (qwen/qwen3.8-27b)...")
    model = create_model()
    judge = GroqJudge(model_name="qwen/qwen3.8-27b", delay_between_calls=1.0)

    faith_metric = FaithfulnessMetric(threshold=0.5, model=judge, include_reason=True)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.5, model=judge, include_reason=True)

    chain_v1 = (
        ChatPromptTemplate.from_messages([
            ("system", ANSWER_PROMPT_V1),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        | model
    )

    chain_v2 = (
        ChatPromptTemplate.from_messages([
            ("system", ANSWER_PROMPT_V2),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])
        | model
    )

    results: List[Dict[str, Any]] = []

    for idx, query in enumerate(TEST_QUESTIONS, start=1):
        print("\n" + "=" * 80)
        print(f"[{idx}/{len(TEST_QUESTIONS)}] Question: {query}")
        print("=" * 80)

        # 1. Retrieve raw candidate chunks
        t0 = time.time()
        retrieved_docs = retriever.invoke(query)
        retrieval_time = time.time() - t0
        raw_chars = sum(len(d.page_content) for d in retrieved_docs)
        print(f"Retrieved 5 chunks ({raw_chars} chars) in {retrieval_time:.2f}s")

        # ----------------------------------------------------
        # 2. Pipeline V1 (Current Compressor V1 + Generator V1)
        # ----------------------------------------------------
        print("\n--- Running V1 (Current Prompt) ---")
        t_v1_start = time.time()
        v1_comp_docs = compress_all(retrieved_docs, query, COMPRESSOR_PROMPT_V1, model)
        v1_comp_time = time.time() - t_v1_start
        v1_chars = sum(len(d.page_content) for d in v1_comp_docs)
        v1_pruned = len(retrieved_docs) - len(v1_comp_docs)
        v1_reduction = ((raw_chars - v1_chars) / raw_chars * 100) if raw_chars > 0 else 0
        v1_context_text = format_docs_with_citations(v1_comp_docs)

        t_v1_gen = time.time()
        v1_resp = chain_v1.invoke({"context": v1_context_text, "chat_history": [], "input": query})
        v1_gen_time = time.time() - t_v1_gen
        v1_answer = v1_resp.content if hasattr(v1_resp, "content") else str(v1_resp)

        print(f"V1: {len(v1_comp_docs)}/5 chunks ({v1_chars} chars, -{v1_reduction:.1f}%), Comp: {v1_comp_time:.2f}s, Gen: {v1_gen_time:.2f}s")
        time.sleep(1.0)

        # ----------------------------------------------------
        # 3. Pipeline V2 (Optimized Compressor V2 + Generator V2)
        # ----------------------------------------------------
        print("\n--- Running V2 (Optimized Prompt) ---")
        t_v2_start = time.time()
        v2_comp_docs = compress_all(retrieved_docs, query, COMPRESSOR_PROMPT_V2, model)
        v2_comp_time = time.time() - t_v2_start
        v2_chars = sum(len(d.page_content) for d in v2_comp_docs)
        v2_pruned = len(retrieved_docs) - len(v2_comp_docs)
        v2_reduction = ((raw_chars - v2_chars) / raw_chars * 100) if raw_chars > 0 else 0
        v2_context_text = format_docs_with_citations(v2_comp_docs)

        t_v2_gen = time.time()
        v2_resp = chain_v2.invoke({"context": v2_context_text, "chat_history": [], "input": query})
        v2_gen_time = time.time() - t_v2_gen
        v2_answer = v2_resp.content if hasattr(v2_resp, "content") else str(v2_resp)

        print(f"V2: {len(v2_comp_docs)}/5 chunks ({v2_chars} chars, -{v2_reduction:.1f}%), Comp: {v2_comp_time:.2f}s, Gen: {v2_gen_time:.2f}s")
        time.sleep(1.0)

        # ----------------------------------------------------
        # 4. Judge Evaluation (Faithfulness & Relevancy)
        # ----------------------------------------------------
        print("Judging V1 output...")
        tc_v1 = LLMTestCase(
            input=query,
            actual_output=v1_answer,
            retrieval_context=[d.page_content for d in v1_comp_docs],
        )
        try:
            faith_metric.measure(tc_v1)
            v1_faith = faith_metric.score
            time.sleep(1.0)
            relevancy_metric.measure(tc_v1)
            v1_rel = relevancy_metric.score
        except Exception as e:
            print(f"Judge error on V1: {e}")
            v1_faith, v1_rel = 1.0, 1.0

        print(f"V1 Judge: Faithfulness={v1_faith:.2f}, Relevancy={v1_rel:.2f}")
        time.sleep(1.0)

        print("Judging V2 output...")
        tc_v2 = LLMTestCase(
            input=query,
            actual_output=v2_answer,
            retrieval_context=[d.page_content for d in v2_comp_docs],
        )
        try:
            faith_metric.measure(tc_v2)
            v2_faith = faith_metric.score
            time.sleep(1.0)
            relevancy_metric.measure(tc_v2)
            v2_rel = relevancy_metric.score
        except Exception as e:
            print(f"Judge error on V2: {e}")
            v2_faith, v2_rel = 1.0, 1.0

        print(f"V2 Judge: Faithfulness={v2_faith:.2f}, Relevancy={v2_rel:.2f}")
        time.sleep(1.0)

        record = {
            "id": idx,
            "question": query,
            "raw_context_chars": raw_chars,
            "v1": {
                "compressed_chars": v1_chars,
                "reduction_pct": round(v1_reduction, 2),
                "chunks_retained": len(v1_comp_docs),
                "chunks_pruned": v1_pruned,
                "comp_time_sec": round(v1_comp_time, 2),
                "gen_time_sec": round(v1_gen_time, 2),
                "total_time_sec": round(retrieval_time + v1_comp_time + v1_gen_time, 2),
                "faithfulness": round(v1_faith, 2),
                "relevancy": round(v1_rel, 2),
                "answer_len": len(v1_answer),
                "has_preamble": "based on" in v1_answer.lower()[:50],
                "answer": v1_answer,
            },
            "v2": {
                "compressed_chars": v2_chars,
                "reduction_pct": round(v2_reduction, 2),
                "chunks_retained": len(v2_comp_docs),
                "chunks_pruned": v2_pruned,
                "comp_time_sec": round(v2_comp_time, 2),
                "gen_time_sec": round(v2_gen_time, 2),
                "total_time_sec": round(retrieval_time + v2_comp_time + v2_gen_time, 2),
                "faithfulness": round(v2_faith, 2),
                "relevancy": round(v2_rel, 2),
                "answer_len": len(v2_answer),
                "has_preamble": "based on" in v2_answer.lower()[:50],
                "answer": v2_answer,
            }
        }
        results.append(record)

    # Compute Averages
    avg_v1_chars = sum(r["v1"]["compressed_chars"] for r in results) / len(results)
    avg_v2_chars = sum(r["v2"]["compressed_chars"] for r in results) / len(results)
    avg_v1_red = sum(r["v1"]["reduction_pct"] for r in results) / len(results)
    avg_v2_red = sum(r["v2"]["reduction_pct"] for r in results) / len(results)
    avg_v1_pruned = sum(r["v1"]["chunks_pruned"] for r in results) / len(results)
    avg_v2_pruned = sum(r["v2"]["chunks_pruned"] for r in results) / len(results)
    avg_v1_gen = sum(r["v1"]["gen_time_sec"] for r in results) / len(results)
    avg_v2_gen = sum(r["v2"]["gen_time_sec"] for r in results) / len(results)
    avg_v1_faith = sum(r["v1"]["faithfulness"] for r in results) / len(results)
    avg_v2_faith = sum(r["v2"]["faithfulness"] for r in results) / len(results)
    avg_v1_rel = sum(r["v1"]["relevancy"] for r in results) / len(results)
    avg_v2_rel = sum(r["v2"]["relevancy"] for r in results) / len(results)

    summary = {
        "v1": {
            "avg_context_chars": round(avg_v1_chars, 1),
            "avg_reduction_pct": round(avg_v1_red, 2),
            "avg_chunks_pruned": round(avg_v1_pruned, 2),
            "avg_gen_time_sec": round(avg_v1_gen, 2),
            "avg_faithfulness": round(avg_v1_faith, 2),
            "avg_relevancy": round(avg_v1_rel, 2),
        },
        "v2": {
            "avg_context_chars": round(avg_v2_chars, 1),
            "avg_reduction_pct": round(avg_v2_red, 2),
            "avg_chunks_pruned": round(avg_v2_pruned, 2),
            "avg_gen_time_sec": round(avg_v2_gen, 2),
            "avg_faithfulness": round(avg_v2_faith, 2),
            "avg_relevancy": round(avg_v2_rel, 2),
        }
    }

    out_file = os.path.join(ROOT_DIR, "eval", "prompt_experiment_results.json")
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "results": results}, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print("PROMPT OPTIMIZATION EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"Average Context Size (Chars) : V1={avg_v1_chars:.1f} -> V2={avg_v2_chars:.1f} (V2 is {((avg_v1_chars-avg_v2_chars)/avg_v1_chars*100):.1f}% more compact)")
    print(f"Average Context Reduction    : V1={avg_v1_red:.1f}% -> V2={avg_v2_red:.1f}%")
    print(f"Average Chunks Pruned        : V1={avg_v1_pruned:.2f} -> V2={avg_v2_pruned:.2f} out of 5")
    print(f"Average Generator Latency    : V1={avg_v1_gen:.2f}s -> V2={avg_v2_gen:.2f}s")
    print(f"Average Faithfulness Score   : V1={avg_v1_faith:.2f} -> V2={avg_v2_faith:.2f}")
    print(f"Average Relevancy Score      : V1={avg_v1_rel:.2f} -> V2={avg_v2_rel:.2f}")
    print(f"Results saved to: {out_file}")
    print("=" * 80)


if __name__ == "__main__":
    run_prompt_experiment()
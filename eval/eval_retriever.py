"""
Retriever Evaluation using DeepEval + Gemini (LLM-as-Judge)

Evaluates the HybridHyDeRetriever against golden test cases using:
  - ContextualRecallMetric    → Did we retrieve content matching expected_output?
  - ContextualPrecisionMetric → Are relevant chunks ranked higher?
  - ContextualRelevancyMetric → Is retrieved context relevant to the query?

Judge: Google Gemini 2.0 Flash (remote API, no local LLM).
"""

import json
import os
import sys

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    ContextualRecallMetric,
    ContextualPrecisionMetric,
    ContextualRelevancyMetric,
)
from deepeval.evaluate import AsyncConfig, DisplayConfig, ErrorConfig

# Add project root to sys.path so imports work when running as a module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from scripts.retriever import build_retriever
from eval.graq_judge import GroqJudge


# ============================================================
# ENV
# ============================================================

load_dotenv()

if not os.getenv("GROQ_API_KEY"):
    raise ValueError("GROQ_API_KEY not found in .env")


# ============================================================
# CONFIG
# ============================================================

GOLDEN_PATH = "goldens/retriever_golden.json"
THRESHOLD = 0.5


# ============================================================
# 1. LOAD GOLDEN SET
# ============================================================

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    goldens = json.load(f)

print(f"\n{'='*50}")
print(f"  Loaded {len(goldens)} golden test cases")
print(f"  Golden file: {GOLDEN_PATH}")
print(f"{'='*50}\n")


# ============================================================
# 2. BUILD RETRIEVER
# ============================================================

print("Building retriever (loading index + BM25 + CrossEncoder)...\n")

# build_retriever() returns (retriever, metrics) -- unpack correctly
# Full pipeline: HyDE (qwen3.8-27b) -> FAISS+BM25 -> CrossEncoder rerank -> top_k
retriever, indexing_metrics = build_retriever()

print(f"Retriever ready. Indexed {indexing_metrics.get('chunks', '?')} chunks.\n")


# ============================================================
# 3. RUN RETRIEVER ON EACH GOLDEN CASE
# ============================================================

test_cases = []

for i, g in enumerate(goldens, start=1):

    print(f"  [{i}/{len(goldens)}] Retrieving for: {g['input'][:60]}...")

    retrieved_docs = retriever.invoke(g["input"])

    retrieval_context = [
        doc.page_content
        for doc in retrieved_docs
    ]

    print(f"           -> Retrieved {len(retrieval_context)} chunks")

    test_cases.append(
        LLMTestCase(
            input=g["input"],
            expected_output=g["expected_output"],
            retrieval_context=retrieval_context,
            # actual_output is required by DeepEval but not used for
            # retriever-only evaluation — set a placeholder
            actual_output="[retriever-only evaluation]",
        )
    )


print(f"\n{'='*50}")
print(f"  Retrieval complete: {len(test_cases)} test cases")
print(f"{'='*50}\n")


# ============================================================
# 4. CREATE LLM JUDGE (remote Groq, no local LLM)
# ============================================================

print("Initializing Groq judge (qwen/qwen3.8-27b)...\n")

judge = GroqJudge(
    model_name="qwen/qwen3.8-27b",
    delay_between_calls=1.0,
    max_retries=3,
)


# ============================================================
# 5. DEFINE METRICS
# ============================================================

metrics = [

    ContextualRecallMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),

    ContextualPrecisionMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),

    ContextualRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
]

print(f"Metrics: {', '.join(m.__class__.__name__ for m in metrics)}")
print(f"Threshold: {THRESHOLD}")
print(f"Judge model: {judge.get_model_name()}\n")


# ============================================================
# 6. EVALUATE (synchronous, throttled)
# ============================================================

print("Starting evaluation (this may take a few minutes)...\n")

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=False,       # Run sequentially to avoid rate limits
        max_concurrent=1,      # One call at a time
        throttle_value=10,     # Extra pause between batches
    ),
    display_config=DisplayConfig(
        print_results=True,
        verbose_mode=None,
    ),
    error_config=ErrorConfig(
        ignore_errors=True,    # Don't crash on individual metric failures
        skip_on_missing_params=True,
    ),
    hyperparameters={
        "retriever": "HybridHyDeRetriever (HyDE: qwen3.8-27b + FAISS/BM25 + CrossEncoder: ms-marco-MiniLM-L-6-v2)",
        "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
        "rerank_model": "cross-encoder/ms-marco-MiniLM-L-6-v2",
        "chunk_size": 500,
        "chunk_overlap": 50,
        "top_k": 5,
        "judge_model": judge.get_model_name(),
        "golden_set": GOLDEN_PATH,
    },
)


# ============================================================
# 7. PRINT SUMMARY
# ============================================================

print(f"\n{'='*60}")
print("  RETRIEVER EVALUATION SUMMARY")
print(f"{'='*60}\n")

for tc in test_cases:
    print(f"Query: {tc.input[:70]}...")
    for metric in metrics:
        # Each metric stores its last score after evaluate()
        # We re-read from the test case's metric data
        pass
    print()

print(f"{'='*60}")
print("  Evaluation complete! Check the table above for per-case scores.")
print(f"{'='*60}\n")
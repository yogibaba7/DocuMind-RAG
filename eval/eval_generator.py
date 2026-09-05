"""
Generator Evaluation using DeepEval (LLM-as-Judge)

Evaluates the generator/QA chain against the golden dataset (generator_golden.json)
using:
  - FaithfulnessMetric  -> Are facts in actual_output strictly grounded in retrieval_context?
  - AnswerRelevancyMetric -> Does actual_output directly answer the input question?

Generator Model: Groq qwen/qwen3.8-27b (bounded max_tokens=350 to prevent OTPM limit)
Judge Model: Groq qwen/qwen3.8-27b via GroqJudge
"""

import json
import os
import sys
import time

from dotenv import load_dotenv

from deepeval import evaluate
from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    FaithfulnessMetric,
    AnswerRelevancyMetric,
)
from deepeval.evaluate import AsyncConfig, DisplayConfig, ErrorConfig

# Add project root to sys.path so imports work properly
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from scripts.rag_chain import ANSWER_SYSTEM_PROMPT
from eval.graq_judge import GroqJudge


# ============================================================
# ENV & CREDENTIALS
# ============================================================

load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")
if not groq_api_key:
    raise ValueError("GROQ_API_KEY not found in .env")


# ============================================================
# CONFIGURATION
# ============================================================

GOLDEN_PATH = "goldens/generator_golden.json"
THRESHOLD = 0.5
GENERATOR_MODEL_NAME = "qwen/qwen3.8-27b"
JUDGE_MODEL_NAME = "qwen/qwen3.8-27b"


# ============================================================
# 1. LOAD GENERATOR GOLDEN DATASET
# ============================================================

with open(GOLDEN_PATH, "r", encoding="utf-8") as f:
    goldens = json.load(f)

print(f"\n{'='*60}")
print(f"  Loaded {len(goldens)} golden cases for generator evaluation")
print(f"  Golden file: {GOLDEN_PATH}")
print(f"{'='*60}\n")


# ============================================================
# 2. INITIALIZE GENERATOR CHAIN
# ============================================================

print(f"Initializing Generator Model ({GENERATOR_MODEL_NAME})...\n")

generator_llm = ChatGroq(
    model=GENERATOR_MODEL_NAME,
    temperature=0,
    max_tokens=350,  # Bounded to stay well under output tokens per minute (OTPM) rate limits
    groq_api_key=groq_api_key,
)

qa_prompt = ChatPromptTemplate.from_messages([
    ("system", ANSWER_SYSTEM_PROMPT),
    ("human", "{input}")
])

qa_chain = qa_prompt | generator_llm | StrOutputParser()


# ============================================================
# 3. RUN GENERATION ON EACH GOLDEN CASE (WITH IDEAL CONTEXT)
# ============================================================

test_cases = []

print("Running generator with ideal context chunks...\n")

for i, g in enumerate(goldens, start=1):
    user_input = g["input"]
    expected_output = g["expected_output"]
    ideal_contexts = g["ideal_context"]

    # Combine ideal context chunks as document context
    combined_context_text = "\n\n---\n\n".join(ideal_contexts)

    print(f"  [{i}/{len(goldens)}] Generating answer for: {user_input[:65]}...")

    try:
        actual_output = qa_chain.invoke({
            "context": combined_context_text,
            "chat_history": [],
            "input": user_input,
        }).strip()
    except Exception as e:
        print(f"     [Warning] Generation failed for case {i}: {e}. Retrying after 2s...")
        time.sleep(2)
        actual_output = qa_chain.invoke({
            "context": combined_context_text,
            "chat_history": [],
            "input": user_input,
        }).strip()

    print(f"           -> Generated: {actual_output[:80]}...\n")

    test_cases.append(
        LLMTestCase(
            input=user_input,
            actual_output=actual_output,
            expected_output=expected_output,
            retrieval_context=ideal_contexts,
        )
    )

    # Slight pause to respect per-minute request rate limits
    time.sleep(0.5)


print(f"\n{'='*60}")
print(f"  Generation complete: {len(test_cases)} test cases ready for judging")
print(f"{'='*60}\n")


# ============================================================
# 4. INITIALIZE LLM JUDGE (Remote Groq, No local LLM)
# ============================================================

print(f"Initializing Groq Judge ({JUDGE_MODEL_NAME})...\n")

judge = GroqJudge(
    model_name=JUDGE_MODEL_NAME,
    delay_between_calls=1.0,
    max_retries=3,
)


# ============================================================
# 5. DEFINE EVALUATION METRICS
# ============================================================

metrics = [
    FaithfulnessMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
    AnswerRelevancyMetric(
        threshold=THRESHOLD,
        model=judge,
        include_reason=True,
    ),
]

print(f"Metrics: {', '.join(m.__class__.__name__ for m in metrics)}")
print(f"Threshold: {THRESHOLD}")
print(f"Judge model: {judge.get_model_name()}\n")


# ============================================================
# 6. RUN EVALUATION
# ============================================================

print("Starting evaluation with DeepEval (this may take a few minutes)...\n")

results = evaluate(
    test_cases=test_cases,
    metrics=metrics,
    async_config=AsyncConfig(
        run_async=False,       # Sequential evaluation to avoid rate limit spikes
        max_concurrent=1,      # 1 call at a time
        throttle_value=1,      # Cooldown between evaluations
    ),
    display_config=DisplayConfig(
        print_results=True,
        verbose_mode=None,
    ),
    error_config=ErrorConfig(
        ignore_errors=True,    # Prevent total script abort on isolated judge errors
        skip_on_missing_params=True,
    ),
    hyperparameters={
        "generator_model": GENERATOR_MODEL_NAME,
        "max_tokens": 350,
        "temperature": 0,
        "judge_model": judge.get_model_name(),
        "golden_set": GOLDEN_PATH,
        "metrics_evaluated": "Faithfulness, AnswerRelevancy",
    },
)


# ============================================================
# 7. SUMMARY
# ============================================================

print(f"\n{'='*60}")
print("  GENERATOR EVALUATION SUMMARY")
print(f"{'='*60}\n")

for idx, tc in enumerate(test_cases, start=1):
    print(f"[{idx}] Question: {tc.input}")
    print(f"    Expected : {tc.expected_output[:90]}...")
    print(f"    Generated: {tc.actual_output[:90]}...\n")

print(f"{'='*60}")
print("  Evaluation complete! Inspect the detailed metrics table above.")
print(f"{'='*60}\n")

import os
from typing import Optional, List, Any
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_groq import ChatGroq
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# ============================================================
# SUPPORTED MODELS LIST (GROQ & HUGGING FACE)
# ============================================================

DEFAULT_MODEL = "qwen/qwen3.8-27b"

GROQ_MODELS: List[str] = [
    "qwen/qwen3.8-27b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

HF_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]

AVAILABLE_MODELS: List[str] = GROQ_MODELS + HF_MODELS


def get_available_models() -> List[str]:
    """Returns the list of available models."""
    return AVAILABLE_MODELS


# ============================================================
# MODEL CREATION FACTORY
# ============================================================

def create_model(
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.1,
    max_tokens: int = 1024,
    streaming: bool = True,
    **kwargs: Any,
) -> BaseChatModel:
    """
    Initializes and returns a chat model via Groq or Hugging Face.
    Prioritizes Groq for speed and reliability, with seamless fallback.
    """
    groq_api_key = os.getenv("GROQ_API_KEY")
    hf_token = os.getenv("HUGGINGFACE_API_KEY")

    # If model is a Groq model or Groq key is present and model is not specifically HF
    is_hf_model = model_name in HF_MODELS
    if groq_api_key and (not is_hf_model or not hf_token):
        actual_model = model_name if model_name in GROQ_MODELS else "qwen/qwen3.8-27b"
        return ChatGroq(
            model=actual_model,
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            groq_api_key=groq_api_key,
            **kwargs,
        )

    # Hugging Face endpoint
    if hf_token:
        try:
            endpoint_llm = HuggingFaceEndpoint(
                repo_id=model_name,
                task="text-generation",
                huggingfacehub_api_token=hf_token,
                max_new_tokens=max_tokens,
                temperature=temperature,
                streaming=streaming,
                **kwargs,
            )
            return ChatHuggingFace(llm=endpoint_llm)
        except Exception as e:
            if groq_api_key:
                # Graceful fallback to Groq if Hugging Face quota is exhausted (402/429)
                return ChatGroq(
                    model="qwen/qwen3.8-27b",
                    temperature=temperature,
                    max_tokens=max_tokens,
                    streaming=streaming,
                    groq_api_key=groq_api_key,
                    **kwargs,
                )
            raise e

    if groq_api_key:
        return ChatGroq(
            model="qwen/qwen3.8-27b",
            temperature=temperature,
            max_tokens=max_tokens,
            streaming=streaming,
            groq_api_key=groq_api_key,
            **kwargs,
        )

    raise ValueError("Neither GROQ_API_KEY nor HUGGINGFACE_API_KEY was found in environment.")


if __name__ == "__main__":
    print(f"Testing Hugging Face model: {DEFAULT_MODEL} ...")
    try:
        model = create_model(model_name=DEFAULT_MODEL)
        print("Model initialized successfully!")
        res = model.invoke("Say 'RAG pipeline with Hugging Face is ready!' in one sentence.")
        print(f"Response:\n{res.content}")
    except Exception as e:
        print(f"Error during test: {e}")

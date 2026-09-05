import os
from typing import Optional, List, Any
from dotenv import load_dotenv

from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace
from langchain_core.language_models.chat_models import BaseChatModel

load_dotenv()

# ============================================================
# HUGGING FACE MODELS LIST
# Easily add or remove model repository IDs here:
# ============================================================

DEFAULT_MODEL = "openai/gpt-oss-120b"

AVAILABLE_MODELS: List[str] = [
    "openai/gpt-oss-120b",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "HuggingFaceH4/zephyr-7b-beta",
]


def get_available_models() -> List[str]:
    """Returns the list of available Hugging Face models."""
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
    Initializes and returns a Hugging Face chat model via HuggingFaceEndpoint.

    Args:
        model_name: Hugging Face repo ID (e.g., 'openai/gpt-oss-120b')
        temperature: Sampling temperature (0.0 to 1.0)
        max_tokens: Maximum new tokens generated
        streaming: Enable token streaming
    """
    hf_token = os.getenv("HUGGINGFACE_API_KEY")
    if not hf_token:
        raise ValueError(
            "HUGGINGFACE_API_KEY not found in .env file. "
            "Please add HUGGINGFACE_API_KEY to your .env file."
        )

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


if __name__ == "__main__":
    print(f"Testing Hugging Face model: {DEFAULT_MODEL} ...")
    try:
        model = create_model(model_name=DEFAULT_MODEL)
        print("Model initialized successfully!")
        res = model.invoke("Say 'RAG pipeline with Hugging Face is ready!' in one sentence.")
        print(f"Response:\n{res.content}")
    except Exception as e:
        print(f"Error during test: {e}")

import os
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEndpoint, ChatHuggingFace

load_dotenv()


def create_model():
    llm = HuggingFaceEndpoint(
                repo_id="openai/gpt-oss-120b",
                task="text-generation",
                huggingfacehub_api_token=os.getenv("HUGGINGFACE_API_KEY"),
                max_new_tokens=512,
                temperature=0.2,
            )

    return ChatHuggingFace(llm=llm)
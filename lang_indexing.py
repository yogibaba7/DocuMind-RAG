import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv()


def create_vector_store(document_path):
    print("Wait Processing Your Document..........")

    # 1. Load document
    loader = PyPDFLoader(document_path)
    documents = loader.load()

    # 2. Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    # 3. Embedding model
    embeddings = HuggingFaceEndpointEmbeddings(
        model="BAAI/bge-m3",
        huggingfacehub_api_token=os.getenv(
            "HUGGINGFACE_API_KEY"
        )
    )

    # 4. Create FAISS vector store
    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    return vector_store
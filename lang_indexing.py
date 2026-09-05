import os
import time
import csv
from datetime import datetime

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_community.vectorstores import FAISS

# improvement
from langchain_community.document_loaders import PyMuPDFLoader

load_dotenv()

METRICS_FILE = "indexing_metrics.csv"


def save_metrics(metrics):

    file_exists = os.path.exists(METRICS_FILE)

    fieldnames = list(metrics.keys())

    # --------------------------------
    # If CSV doesn't exist
    # --------------------------------

    if not file_exists:

        with open(
            METRICS_FILE,
            "w",
            newline="",
            encoding="utf-8"
        ) as file:

            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames
            )

            writer.writeheader()
            writer.writerow(metrics)

        return

    # --------------------------------
    # If CSV already exists
    # --------------------------------

    with open(
        METRICS_FILE,
        "r",
        newline="",
        encoding="utf-8"
    ) as file:

        reader = csv.DictReader(file)

        old_fieldnames = reader.fieldnames or []

        old_records = list(reader)

    # --------------------------------
    # Add new columns
    # --------------------------------

    for field in fieldnames:

        if field not in old_fieldnames:
            old_fieldnames.append(field)

    # --------------------------------
    # Rewrite CSV with new columns
    # --------------------------------

    with open(
        METRICS_FILE,
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=old_fieldnames
        )

        writer.writeheader()

        # Previous records
        for record in old_records:
            writer.writerow(record)

        # New record
        writer.writerow(metrics)


def create_vector_store(document_path="data\sample_policies.pdf"):

    print("Wait Processing Your Document..........")

    # --------------------------------
    # Total timer
    # --------------------------------

    total_start = time.perf_counter()

    # --------------------------------
    # Document Information
    # --------------------------------

    document_name = os.path.basename(document_path)

    document_size_bytes = os.path.getsize(document_path)

    document_size_mb = document_size_bytes / (1024 * 1024)

    # --------------------------------
    # 1. Load Document
    # --------------------------------

    start = time.perf_counter()

    #loader = PyPDFLoader(document_path)
    loader = PyMuPDFLoader(document_path)

    documents = loader.load()

    loading_time = time.perf_counter() - start

    # MB processed per second
    loading_mb_per_sec =  loading_time / document_size_mb 

    print(
        f"Document Loading: "
        f"{loading_time:.2f} sec "
        f"({loading_mb_per_sec:.2f} MB/s)"
    )

    # --------------------------------
    # 2. Split Document
    # --------------------------------

    start = time.perf_counter()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(documents)

    chunking_time = time.perf_counter() - start

    # MB processed per second
    chunking_mb_per_sec =  chunking_time / document_size_mb 

    print(
        f"Chunking: "
        f"{chunking_time:.2f} sec "
        f"({chunking_mb_per_sec:.2f} MB/s)"
    )

    # --------------------------------
    # 3. Create Embedding Model
    # --------------------------------

    embedding_start = time.perf_counter()

    # model="BAAI/bge-m3",
    embeddings = HuggingFaceEndpointEmbeddings(
        model="sentence-transformers/all-MiniLM-L6-v2",
        huggingfacehub_api_token=os.getenv(
            "HUGGINGFACE_API_KEY"
        )
    )

    # --------------------------------
    # 4. Create FAISS Vector Store
    # --------------------------------

    vector_store = FAISS.from_documents(
        chunks,
        embeddings
    )

    embedding_time = time.perf_counter() - embedding_start

    # MB processed per second
    embedding_mb_per_sec =  embedding_time / document_size_mb 

    print(
        f"Embedding + FAISS: "
        f"{embedding_time:.2f} sec "
        f"({embedding_mb_per_sec:.2f} MB/s)"
    )

    # --------------------------------
    # 5. Total Processing Time
    # --------------------------------

    total_time = time.perf_counter() - total_start

    # Total MB processed per second
    total_mb_per_sec = total_time/ document_size_mb 

    print(
        f"Total Processing: "
        f"{total_time:.2f} sec "
        f"({total_mb_per_sec:.2f} MB/s)"
    )

    # --------------------------------
    # 6. Store Metrics
    # --------------------------------

    metrics = {

        "timestamp": datetime.now().isoformat(),

        "document_name": document_name,

        "document_size_mb": round(
            document_size_mb,
            4
        ),

        "pages": len(documents),

        "chunks": len(chunks),

        "chunk_size": 1000,

        "chunk_overlap": 200,

        # Loading
        "loading_time_sec": round(
            loading_time,
            4
        ),

        "loading_mb_per_sec": round(
            loading_mb_per_sec,
            4
        ),

        # Chunking
        "chunking_time_sec": round(
            chunking_time,
            4
        ),

        "chunking_mb_per_sec": round(
            chunking_mb_per_sec,
            4
        ),

        # Embedding
        "embedding_time_sec": round(
            embedding_time,
            4
        ),

        "embedding_mb_per_sec": round(
            embedding_mb_per_sec,
            4
        ),

        # Total
        "total_time_sec": round(
            total_time,
            4
        ),

        "total_mb_per_sec": round(
            total_mb_per_sec,
            4
        )
    }

    # --------------------------------
    # Save Record
    # --------------------------------

    save_metrics(metrics)

    # --------------------------------
    # Display Summary
    # --------------------------------

    print("\n========== INDEXING COMPLETE ==========")

    print(f"Document       : {document_name}")

    print(
        f"Size           : "
        f"{document_size_mb:.2f} MB"
    )

    print(
        f"Pages          : "
        f"{len(documents)}"
    )

    print(
        f"Chunks         : "
        f"{len(chunks)}"
    )

    print(
        f"Loading        : "
        f"{loading_time:.2f} sec "
        f"({loading_mb_per_sec:.2f} MB/s)"
    )

    print(
        f"Chunking       : "
        f"{chunking_time:.2f} sec "
        f"({chunking_mb_per_sec:.2f} MB/s)"
    )

    print(
        f"Embedding      : "
        f"{embedding_time:.2f} sec "
        f"({embedding_mb_per_sec:.2f} MB/s)"
    )

    print(
        f"Total          : "
        f"{total_time:.2f} sec "
        f"({total_mb_per_sec:.2f} MB/s)"
    )

    print("========================================")

    return vector_store

def build_retriever():
    return create_vector_store().as_retriever(search_kwargs={"k":5})

if __name__=="__main__":
    doc_path = "data\IQ_py.pdf"
    vector_store = create_vector_store(document_path=doc_path)
    retriever = build_retriever(vector_store=vector_store)
    result = retriever.invoke("whats is python?")
    for r in result:
        print(f"Page_Number:{r.metadata['page']} -> PageContent:{r.page_content}")
# 📄 DocuMind-RAG

> An AI-powered Document Intelligence system built with LangChain and Retrieval-Augmented Generation (RAG).

DocuMind-RAG allows users to upload a PDF document and ask questions about its content. The system retrieves the most relevant sections from the document and uses a Large Language Model (LLM) to generate grounded answers.

The project is being developed as a modular Document Intelligence platform, with additional capabilities planned beyond question answering.

---

## 🚀 Current Functionality

### 💬 Document Q&A

Users can:

- Upload a PDF document
- Ask questions about the document
- Ask follow-up questions using conversation history
- Get answers grounded in the uploaded document
- Use conversational RAG for context-aware questions

Example:

```text
User:
What is LangChain?

AI:
LangChain is ...

User:
What are its main components?

AI:
The main components are ...
```
🧠 Architecture
```text
The current system follows a conversational RAG architecture:
                    PDF
                     │
                     ▼
              PyPDFLoader
                     │
                     ▼
        RecursiveCharacterTextSplitter
                     │
                     ▼
          Hugging Face Embeddings
                     │
                     ▼
                    FAISS
                     │
                     ▼
                 Retriever
                     ▲
                     │
        Chat History + Question
                     │
                     ▼
          Query Contextualization
                     │
                     ▼
           Standalone Question
                     │
                     ▼
                 Retriever
                     │
                     ▼
            Relevant Documents
                     │
                     ▼
           Context + Chat History
                     │
                     ▼
              PromptTemplate
                     │
                     ▼
             ChatHuggingFace
                     │
                     ▼
             StrOutputParser
                     │
                     ▼
                  Answer
```

🛠️ Tech Stack
```text
Framework
Python
Streamlit
LangChain
Document Processing
PyPDFLoader
RecursiveCharacterTextSplitter
Embeddings
Hugging Face Embedding Models
Vector Database
FAISS
LLM
Hugging Face Chat Models
RAG
Similarity Retrieval
Conversational Query Contextualization
LCEL (LangChain Expression Language)
```

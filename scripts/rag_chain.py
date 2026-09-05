import time
from typing import List, Dict, Any, Generator, Tuple, Optional
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.retrievers import BaseRetriever
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage


# ============================================================
# DOCUMENT FORMATTING & CITATIONS
# ============================================================

def format_docs_with_citations(documents: List[Document]) -> str:
    """
    Formats retrieved document chunks with clear source page annotations
    to allow the LLM and user to inspect exact references.
    """
    if not documents:
        return "No relevant document context found."

    formatted_blocks = []
    for idx, doc in enumerate(documents, start=1):
        page_num = doc.metadata.get("page_number", doc.metadata.get("page", 0) + 1)
        source_name = doc.metadata.get("source_file", doc.metadata.get("source", "Document"))
        content = doc.page_content.strip()

        block = f"[Source {idx} | Document: {source_name} | Page: {page_num}]\n{content}"
        formatted_blocks.append(block)

    return "\n\n---\n\n".join(formatted_blocks)


def format_docs_simple(documents: List[Document]) -> str:
    """Simple newline joining of document content."""
    return "\n\n".join(doc.page_content for doc in documents)


# ============================================================
# CONTEXT COMPRESSION & EXTRACTION
# ============================================================

COMPRESSOR_PROMPT = """You are a context extraction system used in a retrieval pipeline.

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


def compress_document(
    doc: Document,
    query: str,
    extractor_model: BaseChatModel,
) -> Optional[Document]:
    """
    Extracts relevant verbatim spans from a single retrieved document chunk using the extractor LLM.
    If no relevant information is found, returns None to prune the chunk.
    """
    prompt = ChatPromptTemplate.from_template(COMPRESSOR_PROMPT)
    chain = prompt | extractor_model | StrOutputParser()

    extracted = None
    for attempt in range(3):
        try:
            extracted = chain.invoke({
                "query": query,
                "document": doc.page_content,
            }).strip()
            break
        except Exception as e:
            err_msg = str(e)
            if "429" in err_msg or "rate" in err_msg.lower() or attempt < 2:
                time.sleep(2.0 * (attempt + 1))
                continue
            # Fallback gracefully to original document chunk if compression fails
            return doc

    if extracted is None:
        return doc

    # Check for NO_RELEVANT_CONTEXT flag or empty output
    if not extracted or "NO_RELEVANT_CONTEXT" in extracted.upper():
        return None

    # Return compressed Document preserving original metadata (e.g. source, page_number)
    compressed_doc = Document(
        page_content=extracted,
        metadata=dict(doc.metadata),
    )
    compressed_doc.metadata["compressed"] = True
    return compressed_doc


def compress_documents(
    documents: List[Document],
    query: str,
    extractor_model: BaseChatModel,
) -> List[Document]:
    """
    Compresses all retrieved document chunks.
    Filters out uninformative chunks and compresses relevant passages to tightly-scoped spans.
    """
    compressed_list: List[Document] = []
    for doc in documents:
        comp_doc = compress_document(doc, query, extractor_model)
        if comp_doc is not None:
            compressed_list.append(comp_doc)

    # If compression pruned all documents, fallback to the original list so the model has context
    return compressed_list if compressed_list else documents


# ============================================================
# PROMPT DEFINITIONS
# ============================================================

CONTEXTUALIZE_SYSTEM_PROMPT = """Given the conversation history and the latest user question, \
rewrite the latest question as a standalone question that can be understood \
without the previous conversation context.

Rules:
- Do NOT answer the question.
- Only return the rewritten standalone question.
- If the question is already standalone and does not rely on prior context, return it as is.
- Keep the phrasing concise and factual."""

ANSWER_SYSTEM_PROMPT = """You are a highly capable and precise document question-answering assistant.

Your task is to answer the user's question accurately and concisely using ONLY the provided document context below.

Guidelines:
1. Ground your response strictly in the provided context. Do NOT speculate or make up information.
2. If the answer cannot be found or deduced from the provided context, respond politely with:
   "I could not find the answer in the provided document."
3. When helpful, cite the relevant page numbers from the context (e.g., [Page X]).
4. Maintain a clear, professional, and well-structured formatting (use markdown, bullet points, or tables when appropriate).

Context:
{context}"""


# ============================================================
# RAG CHAIN FACTORY
# ============================================================

def create_contextualize_chain(model: BaseChatModel):
    """Creates a chain to rephrase conversational queries into standalone queries."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", CONTEXTUALIZE_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return prompt | model | StrOutputParser()


def create_answer_chain(model: BaseChatModel):
    """Creates the final question-answering chain with chat history support."""
    prompt = ChatPromptTemplate.from_messages([
        ("system", ANSWER_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    return prompt | model | StrOutputParser()


def create_rag_chain(
    retriever: BaseRetriever,
    model: BaseChatModel,
    use_compression: bool = True,
    extractor_model: Optional[BaseChatModel] = None,
):
    """
    Builds an end-to-end conversational RAG chain adhering to LangChain LCEL architecture.
    Optionally compresses retrieved documents using the generator / extractor model.
    """
    contextualize_chain = create_contextualize_chain(model)
    compression_llm = extractor_model if extractor_model is not None else model

    def retrieve_context(data: Dict[str, Any]) -> List[Document]:
        chat_history = data.get("chat_history", [])
        user_input = data.get("input", "")

        # If there is chat history, contextualize the query first
        if chat_history and len(chat_history) > 0:
            try:
                standalone_query = contextualize_chain.invoke({
                    "chat_history": chat_history,
                    "input": user_input,
                }).strip()
            except Exception:
                standalone_query = user_input
        else:
            standalone_query = user_input

        # 1. Retrieve documents
        docs = retriever.invoke(standalone_query)

        # 2. Contextual compression using generator LLM as context extractor
        if use_compression and docs:
            docs = compress_documents(docs, standalone_query, compression_llm)

        return docs

    rag_chain = (
        {
            "context": RunnableLambda(retrieve_context) | format_docs_with_citations,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x.get("chat_history", []),
        }
        | create_answer_chain(model)
    )

    return rag_chain


# ============================================================
# CONVERSATIONAL RAG ENGINE WITH SOURCE CITATIONS & STREAMING
# ============================================================

class ConversationalRAGEngine:
    """
    Encapsulates the conversational RAG lifecycle:
    - Rephrasing queries
    - Retrieving documents
    - Compressing contexts via verbatim span extractor LLM
    - Streaming generation for interactive UI
    """
    def __init__(
        self,
        retriever: BaseRetriever,
        model: BaseChatModel,
        use_compression: bool = True,
        extractor_model: Optional[BaseChatModel] = None,
    ):
        self.retriever = retriever
        self.model = model
        self.use_compression = use_compression
        self.extractor_model = extractor_model if extractor_model is not None else model
        self.contextualize_chain = create_contextualize_chain(model)
        self.answer_prompt = ChatPromptTemplate.from_messages([
            ("system", ANSWER_SYSTEM_PROMPT),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ])

    def get_standalone_query(self, user_input: str, chat_history: List[BaseMessage]) -> str:
        """Transforms multi-turn queries to standalone questions."""
        if not chat_history:
            return user_input

        try:
            standalone = self.contextualize_chain.invoke({
                "chat_history": chat_history,
                "input": user_input,
            }).strip()
            return standalone if standalone else user_input
        except Exception:
            return user_input

    def retrieve_documents(self, search_query: str) -> List[Document]:
        """
        Retrieves matching context chunks and optionally applies contextual compression.
        """
        docs = self.retriever.invoke(search_query)
        if self.use_compression and docs:
            docs = compress_documents(docs, search_query, self.extractor_model)
        return docs

    def generate_response(
        self, user_input: str, chat_history: List[BaseMessage]
    ) -> Tuple[str, List[Document], str]:
        """
        Synchronously generates the response, returning:
        (answer_text, retrieved_documents, standalone_query)
        """
        standalone_query = self.get_standalone_query(user_input, chat_history)
        docs = self.retrieve_documents(standalone_query)
        formatted_context = format_docs_with_citations(docs)

        messages = self.answer_prompt.invoke({
            "context": formatted_context,
            "chat_history": chat_history,
            "input": user_input,
        })

        response = self.model.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)
        return answer, docs, standalone_query

    def stream_response(
        self, user_input: str, chat_history: List[BaseMessage]
    ) -> Tuple[Generator[str, None, None], List[Document], str]:
        """
        Retrieves context, compresses relevant spans, and returns a token generator
        for real-time UI streaming, along with the compressed documents and standalone query.
        """
        standalone_query = self.get_standalone_query(user_input, chat_history)
        docs = self.retrieve_documents(standalone_query)
        formatted_context = format_docs_with_citations(docs)

        messages = self.answer_prompt.invoke({
            "context": formatted_context,
            "chat_history": chat_history,
            "input": user_input,
        })

        def token_generator():
            for chunk in self.model.stream(messages):
                if hasattr(chunk, "content"):
                    yield chunk.content
                else:
                    yield str(chunk)

        return token_generator(), docs, standalone_query

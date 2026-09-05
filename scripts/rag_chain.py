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


def create_rag_chain(retriever: BaseRetriever, model: BaseChatModel):
    """
    Builds an end-to-end conversational RAG chain adhering to LangChain LCEL architecture.
    """
    contextualize_chain = create_contextualize_chain(model)

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

        # Retrieve documents
        return retriever.invoke(standalone_query)

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
    - Retrieving documents and capturing source metadata
    - Streaming generation for interactive UI
    """
    def __init__(self, retriever: BaseRetriever, model: BaseChatModel):
        self.retriever = retriever
        self.model = model
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
        """Retrieves matching context chunks."""
        return self.retriever.invoke(search_query)

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
        Retrieves context and returns a token generator for real-time UI streaming,
        along with the retrieved documents and standalone query.
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

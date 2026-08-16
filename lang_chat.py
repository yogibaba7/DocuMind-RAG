# from langchain_core.prompts import PromptTemplate
# from langchain_core.runnables import RunnablePassthrough
# from langchain_core.output_parsers import StrOutputParser


# def format_docs(docs):
#     return "\n\n".join(doc.page_content for doc in docs)


# def create_chat_chain(vector_store, model):

#     retriever = vector_store.as_retriever(
#         search_type="similarity",
#         search_kwargs={"k": 4}
#     )

#     prompt = PromptTemplate.from_template("""
# You are a helpful document question-answering assistant.

# Answer the question using ONLY the provided context.

# If the answer cannot be found in the context, say:
# "I could not find the answer in the provided document."

# Do not make up information.

# Context:
# {context}

# Question:
# {question}

# Answer:
# """)

#     chain = (
#         {
#             "context": retriever | format_docs,
#             "question": RunnablePassthrough()
#         }
#         | prompt
#         | model
#         | StrOutputParser()
#     )

#     return chain


# ________________________________________________context Based Chat__________________________________________________________
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


def create_chat_chain(vector_store, model):

    # -------------------------
    # 1. Retriever
    # -------------------------

    retriever = vector_store.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # -------------------------
    # 2. Query contextualization
    # -------------------------

    contextualize_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """Given the chat history and the latest user question,
rewrite the latest question as a standalone question.

Do NOT answer the question.
Only return the rewritten question.

If the question is already standalone, return it unchanged."""
        ),

        MessagesPlaceholder("chat_history"),

        ("human", "{input}")
    ])

    contextualize_chain = (
        contextualize_prompt
        | model
        | StrOutputParser()
    )

    # -------------------------
    # 3. Retrieve documents
    # -------------------------

    def retrieve_with_history(data):

        standalone_question = contextualize_chain.invoke({
            "chat_history": data["chat_history"],
            "input": data["input"]
        })

        documents = retriever.invoke(standalone_question)

        return documents

    # -------------------------
    # 4. Format documents
    # -------------------------

    def format_docs(documents):
        return "\n\n".join(
            document.page_content
            for document in documents
        )

    # -------------------------
    # 5. Final answer prompt
    # -------------------------

    answer_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a helpful document question-answering assistant.

Answer the user's question using ONLY the provided document context.

If the answer cannot be found in the context, say:
"I could not find the answer in the provided document."

Do not make up information.

Context:
{context}"""
        ),

        MessagesPlaceholder("chat_history"),

        ("human", "{input}")
    ])

    # -------------------------
    # 6. Complete RAG chain
    # -------------------------

    chain = (
        {
            "context": RunnableLambda(retrieve_with_history) | format_docs,
            "input": lambda x: x["input"],
            "chat_history": lambda x: x["chat_history"]
        }
        | answer_prompt
        | model
        | StrOutputParser()
    )

    return chain
from lang_indexing import create_vector_store
from lang_chat import create_chat_chain
from lang_generation import create_model
from langchain_core.messages import HumanMessage, AIMessage

import streamlit as st

st.set_page_config(
    page_title="Document Q&A",
    page_icon="📄"
)

st.title("📄 Document Q&A")

# Initialize session state
if "chat_chain" not in st.session_state:
    st.session_state.chat_chain = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# -------------------------
# Document Upload
# -------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)


if uploaded_file:

    if st.session_state.chat_chain is None:

        with st.spinner("Processing document..."):

            # Save uploaded file temporarily
            with open("temp.pdf", "wb") as f:
                f.write(uploaded_file.getbuffer())

            # Create vector store
            vector_store = create_vector_store("temp.pdf")

            # Create LLM
            model = create_model()

            # Create RAG chain
            st.session_state.chat_chain = create_chat_chain(
                vector_store,
                model
            )

        st.success("Document processed successfully! Ask your questions below.")


# -------------------------
# Chat Interface
# -------------------------
if st.session_state.chat_chain:

    # Display previous messages
    for chat in st.session_state.chat_history:

        role = "user" if isinstance(chat, HumanMessage) else "assistant"

        with st.chat_message(role):
            st.markdown(chat.content)

    question = st.chat_input(
        "Ask something about the document..."
    )

    if question:

        # Add user message
        st.session_state.chat_history.append(
            HumanMessage(content=question)
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate answer
        with st.chat_message("assistant"):

            with st.spinner("Thinking..."):

                answer = st.session_state.chat_chain.invoke({
                    "input": question,
                    "chat_history": st.session_state.chat_history
                })

            st.markdown(answer)

        # Add assistant message
        st.session_state.chat_history.append(
            AIMessage(content=answer)
        )
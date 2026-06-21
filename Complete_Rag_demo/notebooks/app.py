import os
import streamlit as st
from pathlib import Path

import fitz  # optional (removed from pipeline, kept if you want later)

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_ollama import ChatOllama


# =====================================================
# CONFIG
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

DOCS_PATH = BASE_DIR.parent / "company_docs"
VECTOR_DB = str(BASE_DIR / "vectorstore")

DOC_FOLDER = str(DOCS_PATH)


# =====================================================
# LOAD DOCUMENTS
# =====================================================

def load_documents():
    loader = DirectoryLoader(
        DOC_FOLDER,
        glob="**/*.pdf",
        loader_cls=PyPDFLoader
    )
    return loader.load()


# =====================================================
# VECTOR STORE
# =====================================================

@st.cache_resource
def create_or_load_vectorstore():

    os.makedirs(VECTOR_DB, exist_ok=True)

    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    index_path = os.path.join(VECTOR_DB, "index.faiss")

    if os.path.exists(index_path):
        return FAISS.load_local(
            VECTOR_DB,
            embeddings,
            allow_dangerous_deserialization=True
        )

    st.info("Creating Vector Database... This may take a while ⏳")

    docs = load_documents()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = splitter.split_documents(docs)

    db = FAISS.from_documents(chunks, embeddings)

    db.save_local(VECTOR_DB)

    return db


db = create_or_load_vectorstore()

retriever = db.as_retriever(search_kwargs={"k": 4})


# =====================================================
# LLM (Ollama fallback to OpenAI)
# =====================================================

@st.cache_resource
def load_llm():

    try:
        llm = ChatOllama(model="llama3.3")

        # simple warmup check
        llm.invoke("hello")

        st.sidebar.success("Using Ollama 🦙")

        return llm

    except Exception:
        st.sidebar.warning("Using OpenAI GPT-4o-mini")

        return ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0
        )


llm = load_llm()


# =====================================================
# PROMPT
# =====================================================

prompt = ChatPromptTemplate.from_template(
    """
You are a company knowledge assistant.

Answer ONLY using the context below.

If you don't know, say:
"I could not find that information in the documents."

Context:
{context}

Question:
{question}

Answer:
"""
)


# =====================================================
# FORMAT DOCS
# =====================================================

def format_docs(docs):
    return "\n\n".join(
        f"""
SOURCE: {doc.metadata.get('source', 'unknown')}
PAGE: {doc.metadata.get('page', 'unknown')}

{doc.page_content}
"""
        for doc in docs
    )


# =====================================================
# RAG CHAIN (LCEL)
# =====================================================

rag_chain = (
    {
        "context": retriever | RunnableLambda(format_docs),
        "question": RunnablePassthrough()
    }
    | prompt
    | llm
)


# =====================================================
# STREAMLIT UI
# =====================================================

st.title("📄 Company RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []


# show history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])


# input
question = st.chat_input("Ask something from company documents...")

if question:

    st.session_state.messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking... 🤔"):

            response = rag_chain.invoke(question)
            answer = response.content

            st.markdown(answer)

            docs = retriever.invoke(question)

            with st.expander("📚 Sources"):
                for doc in docs:
                    st.write(
                        f"📄 {doc.metadata.get('source')} | Page {doc.metadata.get('page')}"
                    )

    st.session_state.messages.append(
        {"role": "assistant", "content": answer}
    )

    
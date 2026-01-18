import os
from dotenv import load_dotenv
import streamlit as st
from langchain_groq import ChatGroq
from langchain_classic.chains import RetrievalQAWithSourcesChain
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import FastEmbedEmbeddings

# ---------------- UI ----------------
load_dotenv()
st.set_page_config(page_title="Website-Based AI Chatbot")
st.title("🌐 Website-Based AI Chatbot")

st.sidebar.header("Website URL")
url = st.sidebar.text_input("Enter URL")
process_url_clicked = st.sidebar.button("Process URL")

status = st.empty()

# ---------------- Indexing ----------------
if process_url_clicked:
    if not url:
        st.error("Please enter a valid URL.")
    else:
        try:
            status.text("Loading website...")
            loader = UnstructuredURLLoader(urls=[url])  # ✅ FIXED
            data = loader.load()

            if not data:
                st.error("No content could be extracted from this website.")
                st.stop()

            status.text("Splitting content...")
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200
            )
            docs = splitter.split_documents(data)

            if not docs:
                st.error("Website content is empty after processing.")
                st.stop()

            status.text("Creating embeddings...")
            embeddings = FastEmbedEmbeddings(
                model_name="BAAI/bge-small-en-v1.5"
            )

            vector_index = FAISS.from_documents(docs, embeddings)
            st.session_state.vector_index = vector_index

            status.success("Website indexed successfully!")

        except Exception as e:
            st.error(f"Indexing failed: {e}")
            st.stop()

# ---------------- LLM ----------------
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile",
    temperature=0
)

# ---------------- QA ----------------
query = st.text_input("Ask a question about the website")

if query:
    if "vector_index" not in st.session_state:
        st.error("Please process a website first.")
    else:
        qa_chain = RetrievalQAWithSourcesChain.from_llm(
            llm=llm,
            retriever=st.session_state.vector_index.as_retriever()
        )

        result = qa_chain(
            {"question": query},
            return_only_outputs=True
        )

        st.subheader("Answer")
        st.write(result.get("answer", "The answer is not available on the provided website."))

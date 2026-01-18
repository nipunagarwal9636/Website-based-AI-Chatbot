# 🌐 Website-Based AI Chatbot (Groq + FAISS)

## 1. Project Overview

This project implements a **website-aware AI chatbot** that answers user questions **strictly based on the content of a given website**. The chatbot first indexes the website content and then uses semantic search combined with a large language model (LLM) to generate grounded answers. If the required information is not present on the website, the chatbot explicitly responds with:

> **"The answer is not available on the provided website."**

The goal is to prevent hallucinations and ensure factual, source-based responses.

---

## 2. Architecture Explanation

The application follows a **Retrieval-Augmented Generation (RAG)** architecture:

1. **Website Ingestion**

   * A public URL is provided by the user.
   * The website content is extracted and cleaned.

2. **Text Chunking**

   * Extracted text is split into overlapping chunks to preserve semantic context.

3. **Embedding Generation**

   * Each text chunk is converted into a vector embedding.

4. **Vector Storage**

   * Embeddings are stored in a FAISS vector database for efficient similarity search.

5. **Query Processing**

   * User questions are embedded and matched against the stored vectors.

6. **Answer Generation**

   * Relevant chunks are passed to the LLM, which generates an answer strictly from retrieved context.

7. **Fallback Handling**

   * If no relevant context is found, the chatbot returns a predefined fallback message.

---

## 3. Frameworks Used

* **Streamlit** – Interactive web interface for user interaction
* **LangChain** – Orchestrates document loading, chunking, embeddings, retrieval, and QA chains
* **FAISS** – High-performance vector similarity search
* **FastEmbed** – Lightweight, ONNX-based embedding generation (torch-free)

LangGraph was not used as the workflow does not require complex stateful agent graphs.

---

## 4. LLM Model Used and Rationale

**Model:** `llama-3.x` via **Groq API**

**Why Groq + LLaMA 3?**

* Extremely low latency inference
* High-quality reasoning and summarization
* Suitable for retrieval-augmented question answering
* Reliable hosted inference without local GPU requirements

The model is configured with **temperature = 0** to minimize hallucinations.

---

## 5. Vector Database Used and Rationale

**Vector Database:** FAISS (Facebook AI Similarity Search)

**Why FAISS?**

* Fast and memory-efficient similarity search
* Well-supported by LangChain
* Ideal for small-to-medium scale RAG applications
* Simple local persistence without external services

---

## 6. Embedding Strategy

**Embedding Model:** `BAAI/bge-small-en-v1.5` via FastEmbed

**Strategy:**

* Website content is split into overlapping chunks
* Each chunk is embedded using a semantic embedding model
* User queries are embedded using the same model
* Cosine similarity search retrieves the most relevant chunks

**Why FastEmbed?**

* Torch-free (no PyTorch or CUDA dependencies)
* Stable on Windows
* Lightweight and production-friendly

---

## 7. Setup and Run Instructions

### Prerequisites

* Python 3.10 or 3.11
* A Groq API key

### Installation

```bash
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### Configure API Key

**Local:**

```powershell
setx GROQ_API_KEY "your_groq_api_key"
```

**Streamlit Cloud:**
Add the following to **App Secrets**:

```toml
GROQ_API_KEY = "your_groq_api_key"
```

### Run the App

```bash
streamlit run project.py
```

---

## 8. Assumptions

* The website is publicly accessible
* The website contains extractable textual content
* The website does not block scraping
* Questions are related to the indexed website

---

## 9. Limitations

* JavaScript-heavy websites may not load correctly
* Multi-page crawling is not enabled by default
* No authentication-protected pages supported
* Vector index is in-memory per session

---

## 10. Future Improvements

* Multi-page and sitemap-based crawling
* Persistent vector index storage
* Source citation highlighting in UI
* Improved HTML parsing fallback
* Chat history and conversational memory
* Deployment with Docker

---


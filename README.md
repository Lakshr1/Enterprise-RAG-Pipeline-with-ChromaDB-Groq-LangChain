# 🚀 **Enterprise RAG Pipeline (Lightweight Edition) with ChromaDB + LangChain + Groq**

*A simple, clean, and fully functional Retrieval-Augmented Generation pipeline implemented in Jupyter notebooks.*

---

## 📌 **Overview**

This project demonstrates a **complete, minimal, and production-oriented RAG pipeline** using:

* **ChromaDB** – vector storage
* **Sentence-Transformers** – text embeddings
* **LangChain** – document loaders, chunking, orchestration
* **Groq LPU inference** – ultra-fast LLM responses
* **PDF, Excel, CSV, Text ingestion**
* **RAG with context retrieval + LLM answer generation**
* **Enhanced features including citations, confidence scores, and streaming**

The entire pipeline is implemented using just **two Jupyter Notebooks**, making it easy to understand, reproduce, and extend.

---

## 📁 **Project Structure**

```
📦 enterprise-rag-simple
│
├── data/
│   ├── text_files/       # .txt sample data
│   ├── pdf/              # PDFs for ingestion
│   ├── excel/            # Excel (*.xls) files
│   └── vector_store/     # ChromaDB persistent store
│
├── document.ipynb       # Notebook 1: Document ingestion + parsing
├── loader.ipynb         # Notebook 2: RAG pipeline, embeddings, ChromaDB, Groq LLM
├── requirements.txt
└── README.md
```

---

## 🧠 **What This Project Covers**

### **📌 1. Multi-Format Data Ingestion**

Using LangChain’s loaders:

* `TextLoader` for `.txt`
* `DirectoryLoader` for bulk ingestion
* `PyPDFLoader` / `PyMuPDFLoader` for PDFs
* `UnstructuredExcelLoader` for Excel files

Example:

```python
loader = TextLoader("../data/text_files/Why_RAG.txt")
documents = loader.load()
```

---

### **📌 2. Text Splitting / Chunking**

Using `RecursiveCharacterTextSplitter` for optimal chunk sizes:

```python
splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
chunks = splitter.split_documents(documents)
```

---

### **📌 3. Embeddings with Sentence Transformers**

```python
EmbeddingManager("all-MiniLM-L6-v2")
```

Generates embeddings for each chunk before indexing.

---

### **📌 4. ChromaDB Vector Store**

Full support for:

* Persistent storage
* Metadata saving
* Chunk indexing
* Fast similarity search

Example:

```python
vectorstore = VectorStore("pdf_documents", "../data/vector_store")
vectorstore.add_documents(chunks, embeddings)
```

---

### **📌 5. RAG Retrieval Pipeline**

Custom retriever that:

* Embeds the query
* Retrieves top-K relevant chunks
* Converts cosine distance → similarity score
* Filters by threshold

```python
results = retriever.retrieve("What is attention mechanism?", top_k=3)
```

---

### **📌 6. Groq LPU-Powered LLM Answering**

Using super-fast Groq inference:

```python
llm = ChatGroq(
    groq_api_key=os.getenv("GROQ_API_KEY"),
    model_name="openai/gpt-oss-120b",
    temperature=0.1
)
```

---

### **📌 7. Simple RAG Function**

```python
answer = rag_simple("What is attention?", retriever, llm)
```

---

### **📌 8. Advanced RAG Features**

The notebook includes:

* Answer + source attribution
* Confidence scores
* Context return option
* PDF page-level metadata
* Streaming output simulation
* Summary generation
* Query history tracking
* Clean citation formatting

Example:

```python
result = rag_advanced(
    "Explain attention mechanism",
    retriever,
    llm,
    top_k=5,
    min_score=0.1,
    return_context=True
)
```

---

### **📌 9. Full Advanced Pipeline Class**

A reusable class with:

* Intelligent retrieval
* LLM formatting
* Streaming
* Summaries
* Citations
* History
* JSON-style structured output

```python
adv_rag = AdvancedRAGPipeline(retriever, llm)
result = adv_rag.query("What is Attention?", summarize=True)
```

---

## 🎯 **Example Query Output**

```
Answer:
The attention mechanism allows a model to focus on relevant parts of the input...

Citations:
[1] attention-is-all-you-need.pdf (page 3)
[2] ray-distributed-framework-AI-app.pdf (page 1)

Confidence: 0.87
```

---

## 🛠️ **Installation**

```
pip install -r requirements.txt
```

Environment variable:

```
export GROQ_API_KEY=your_key
```

---

## 🚀 **How to Run**

### Notebook 1 → **document.ipynb**

Covers:

* Creating sample text files
* PDF loader
* Excel loader
* Data ingestion tests
* Document object creation

### Notebook 2 → **loader.ipynb**

Covers:

* Chunking
* Embeddings
* ChromaDB
* Retrieval
* Simple RAG
* Advanced RAG
* Groq LLM responses

---

## 🧩 **Extendability**

This minimal RAG is designed so you can easily extend it into:

* RAG with LlamaIndex
* Multi-Agent pipelines (LangGraph / MPC)
* Fine-tuned RAG (LoRA adapters)
* Streaming UI (Streamlit / FastAPI)
* Structured outputs (JSON mode)
* Domain-specific supply chain RAG

---

## 📄 **Requirements**

* Python 3.10+
* langchain
* langchain-community
* langchain_groq
* chromadb
* sentence-transformers
* PyPDF2, PyMuPDF
* unstructured
* scikit-learn
* numpy

All included in `requirements.txt`.

---

## ⭐ **Why This Project Matters**

Even though it’s simple, this RAG pipeline demonstrates:

* Real **enterprise RAG building blocks**
* Understanding of **embeddings, vector stores, retrieval**
* Low-latency **LLM integration using Groq**
* Document ingestion for **PDF/Excel/CSV/Text**
* Ability to structure clean, modular code
* Knowledge of **advanced RAG evaluation features**

Exactly what companies like Cisco want for **LLM/Data Scientist roles**.

---

## 🤝 **Contributions**

Feel free to open PRs for enhancements like:

* Hybrid retrievers
* Multi-vector embeddings
* Evaluation dashboard
* Streamlit UI

---

## 📬 **Contact**

For any queries, reach out:
📧 **[raolakshrajsingh@gmail.com](mailto:raolakshrajsingh@gmail.com)**

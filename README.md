<div align="center">

# 📄 PDF Q&A Assistant
### LLM-Powered Document Intelligence with RAG

[![Python](https://img.shields.io/badge/Python-3.11-blue?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Groq](https://img.shields.io/badge/Groq-LLaMA_3.1-orange?style=for-the-badge&logo=meta&logoColor=white)](https://groq.com)
[![FAISS](https://img.shields.io/badge/FAISS-Vector_Search-green?style=for-the-badge&logo=facebook&logoColor=white)](https://faiss.ai)
[![License](https://img.shields.io/badge/License-MIT-purple?style=for-the-badge)](LICENSE)

<br/>

> **Upload any PDF. Ask anything. Get intelligent answers — instantly.**

<br/>

![Demo](https://img.shields.io/badge/🚀_Live_Demo-Click_Here-brightgreen?style=for-the-badge)

**[🌐 Try the Live App](https://imadh-ai-pdf-gpt-assistant.streamlit.app)**

<br/>

---

</div>

## ✨ Features

| Feature | Description |
|--------|-------------|
| 📤 **PDF Upload** | Upload any PDF up to 200MB |
| 🧠 **RAG Pipeline** | Retrieval-Augmented Generation for accurate answers |
| ✂️ **Smart Chunking** | Overlapping semantic chunking for better context |
| 🔍 **Vector Search** | FAISS-powered similarity search across document |
| 💬 **Chat Interface** | Clean conversational Q&A with history |
| 📚 **Source Excerpts** | See exactly which parts of the PDF were used |
| 📥 **Transcript Export** | Download your full Q&A session as a `.txt` file |
| 🔄 **Multi-PDF Support** | Switch between PDFs seamlessly |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        USER UPLOADS PDF                      │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  📄 PDF PROCESSOR                                            │
│  pdfplumber → extract text → clean → semantic chunks        │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  🧠 EMBEDDING ENGINE                                         │
│  sentence-transformers (all-MiniLM-L6-v2) → 384-dim vectors │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  🔍 FAISS VECTOR STORE                                       │
│  IndexFlatIP + L2 normalization → cosine similarity search  │
└─────────────────────┬───────────────────────────────────────┘
                      │
              USER ASKS QUESTION
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  🎯 RETRIEVER                                                │
│  Embed query → search FAISS → top-k relevant chunks         │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  🤖 GROQ LLaMA 3.1                                           │
│  Context + Question → Accurate grounded answer              │
└─────────────────────┬───────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────┐
│  💬 STREAMLIT UI                                             │
│  Chat interface + source excerpts + export transcript        │
└─────────────────────────────────────────────────────────────┘
```

---

## 🛠️ Tech Stack

```python
tech_stack = {
    "Frontend":    "Streamlit",
    "LLM":         "LLaMA 3.1 8B via Groq API",
    "Embeddings":  "sentence-transformers (all-MiniLM-L6-v2)",
    "Vector DB":   "FAISS (Facebook AI Similarity Search)",
    "PDF Parser":  "pdfplumber + PyPDF2",
    "Language":    "Python 3.11",
    "Hosting":     "Streamlit Community Cloud",
}
```

---

## 🚀 Getting Started

### Prerequisites
- Python 3.10+
- [Groq API Key](https://console.groq.com) (free)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/Imadh-ai/PDF-GPT-Assistant.git
cd PDF-GPT-Assistant

# 2. Create virtual environment
python3 -m venv venv
source venv/bin/activate  # Mac/Linux
# venv\Scripts\activate   # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment variables
touch .env
```

Add your Groq API key to `.env`:
```env
GROQ_API_KEY=gsk_your-key-here
```

```bash
# 5. Run the app
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 📁 Project Structure

```
PDF-GPT-Assistant/
│
├── app.py                  # 🎨 Streamlit UI — main entry point
├── requirements.txt        # 📦 Python dependencies
├── .env                    # 🔑 API keys (never commit this!)
├── .gitignore
│
├── src/
│   ├── __init__.py
│   ├── pdf_processor.py    # 📄 PDF parsing & semantic chunking
│   ├── embeddings.py       # 🧠 Vector embeddings (local)
│   ├── retriever.py        # 🔍 FAISS vector search
│   └── llm_chain.py        # 🤖 Groq LLaMA RAG pipeline
│
└── utils/
    ├── __init__.py
    └── helpers.py          # 🛠️ Session history & transcript export
```

---

## 💡 How It Works

### 1️⃣ Document Processing
The PDF is parsed using `pdfplumber` (with `PyPDF2` as fallback), cleaned, and split into **500-word overlapping chunks** (50-word overlap) to preserve context at boundaries.

### 2️⃣ Semantic Embedding
Each chunk is converted into a **384-dimensional vector** using `sentence-transformers`. This runs entirely locally — no API calls, no cost, no limits.

### 3️⃣ Vector Indexing
All vectors are stored in a **FAISS index** with L2 normalization for cosine similarity search. This enables lightning-fast semantic search across thousands of chunks.

### 4️⃣ Retrieval-Augmented Generation
When a question is asked:
- The question is embedded into a vector
- FAISS finds the **top-5 most similar chunks**
- Chunks + question are sent to **Groq LLaMA 3.1**
- The model answers **only from the retrieved context**

### 5️⃣ Grounded Answers
The LLM is instructed to answer **only from the document** — it will say "I couldn't find that in the document" rather than hallucinating an answer.

---

## 🎯 Use Cases

- 📚 **Research papers** — extract key findings instantly
- 📑 **Legal documents** — query contracts and agreements
- 📘 **Textbooks** — ask questions about specific chapters
- 📊 **Reports** — summarize and query business reports
- 📝 **Study notes** — interactive Q&A with your own notes

---

## ⚙️ Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `chunk_size` | 500 words | Words per chunk |
| `overlap` | 50 words | Overlap between chunks |
| `top_k` | 5 | Chunks retrieved per query |
| `temperature` | 0.2 | LLM creativity (lower = more factual) |
| `max_tokens` | 1024 | Max response length |

---

## 🙋 Author

<div align="center">

**Imadh**

[![GitHub](https://img.shields.io/badge/GitHub-Imadh--ai-black?style=for-the-badge&logo=github)](https://github.com/Imadh-ai)

*Built with ❤️ using Python, Streamlit, and LLaMA 3.1*

</div>

---

<div align="center">

⭐ **Star this repo if you found it useful!** ⭐

</div>

    Chat-based tool to query and extract answers from PDF documents using AI.

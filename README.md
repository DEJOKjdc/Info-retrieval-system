# Info-retrieval-system
# 📚 Information Retrieval System using RAG

A Retrieval-Augmented Generation (RAG) application that allows users to upload multiple PDF documents and ask questions in natural language. The system retrieves the most relevant information from the uploaded documents using semantic search and generates accurate responses using Google's Gemini LLM.

---

## Features

- Upload multiple PDF documents
- Automatic text extraction
- Intelligent text chunking
- Semantic embeddings using Gemini
- FAISS vector database
- Conversational Question Answering
- Chat history memory
- Streamlit web interface

---

## Tech Stack

| Technology | Purpose |
|------------|---------|
| Python | Backend |
| Streamlit | Web UI |
| LangChain | LLM Framework |
| Gemini 1.5 Flash | Large Language Model |
| embedding-001 | Text Embeddings |
| FAISS | Vector Database |
| PyPDF2 | PDF Processing |
| python-dotenv | Environment Variables |

---

## Project Structure

```
info-retrieval-system/

│
├── app.py
├── setup.py
├── requirements.txt
├── .env
│
├── src/
│   ├── __init__.py
│   └── helper.py
│
├── research/
│   └── trials.ipynb
│
└── test.py
```

---

## Installation

### Clone Repository

```bash
git clone https://github.com/yourusername/info-retrieval-system.git

cd info-retrieval-system
```

### Create Virtual Environment

Windows

```bash
python -m venv venv
venv\Scripts\activate
```

Linux / macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file.

```env
GOOGLE_API_KEY=YOUR_GOOGLE_API_KEY
```

---

## Run the Application

```bash
streamlit run app.py
```

The application will start on

```
http://localhost:8501
```

---

## How it Works

### Step 1

Upload one or more PDF documents.

↓

### Step 2

Text is extracted using PyPDF2.

↓

### Step 3

The extracted text is divided into overlapping chunks using RecursiveCharacterTextSplitter.

↓

### Step 4

Each chunk is converted into a vector embedding using Gemini's `embedding-001` model.

↓

### Step 5

All embeddings are stored inside a FAISS vector database.

↓

### Step 6

When the user asks a question, FAISS retrieves the most relevant chunks.

↓

### Step 7

The retrieved context and the user's question are passed to Gemini 1.5 Flash.

↓

### Step 8

Gemini generates a context-aware answer.

---

## Architecture

```
PDF Files
    │
    ▼
PyPDF2
    │
    ▼
Extract Text
    │
    ▼
Chunking
    │
    ▼
Gemini Embeddings
    │
    ▼
FAISS
    │
User Question
    │
    ▼
Similarity Search
    │
Relevant Chunks
    │
    ▼
Gemini LLM
    │
    ▼
Generated Answer
```

---

## Technologies Used

- Python
- Streamlit
- LangChain
- Google Gemini
- FAISS
- PyPDF2
- dotenv

---

## Future Improvements

- OCR support for scanned PDFs
- Citation and page number references
- Source highlighting
- Hybrid search (BM25 + Vector Search)
- Persistent FAISS storage
- Authentication
- Chat export
- PDF summarization
- Multi-format support (DOCX, PPTX, TXT)
- Streaming responses




# RAG AI/ML Chatbot (Internal Project)

## Description
This repository contains an internal-level AI/ML project implementing a **Retrieval-Augmented Generation (RAG)** based chatbot.  
The objective of this project is to demonstrate backend design, document ingestion, vector-based retrieval, and LLM integration.

This is intended as a **proof-of-concept (POC)** and not a full production deployment.

---

## Project Structure
```bash 
RAG_AIML_CHATBOT/
├── data/                               # Source documents for ingestion
├── src/
│ ├── ingest.py                        # Handles document loading, chunking, and embeddings
│ └── rag.py                           # Core RAG pipeline (retrieval + generation)
├── vectorstore/
│ ├── index.faiss                      # FAISS vector index
│ └── index.pkl                        # Metadata for vector store
├── app.py                             # Application entry point
├── test_rag.py                        # Basic test script for RAG flow
├── requirements.txt                   # Python dependencies
├── runtime.txt                        # Python runtime specification
├── .env                               # Environment variables (ignored in git)
├── .gitignore

```



## Tech Stack
- Python
- FAISS (vector similarity search)
- LLM + Embedding models
- dotenv for environment configuration

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone <repository-url>
cd RAG_AIML_CHATBOT
```


### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```



## Configuration

Create a `.env` file in the project root and add the required API keys and configuration values.

Example:
```env
OPENAI_API_KEY=your_key_here
```


## Run the Project

### Step 1: Ingest documents
```bash
python src/ingest.py
```

### Step 2: Run the application
```bash
python app.py
```

### Notes

This project is designed for internal evaluation and learning purposes

Vector storage is file-based and local

A simple UI is included for interacting with the chatbot

Deployment and scalability concerns are intentionally out of scope



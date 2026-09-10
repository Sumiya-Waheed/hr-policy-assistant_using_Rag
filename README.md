# 📘 HR Policy Assistant using RAG

An AI-powered HR Policy Assistant built with Retrieval-Augmented Generation (RAG).

Users can upload an HR Policy PDF and ask questions about the document. The application retrieves the most relevant sections from the uploaded policy and uses Groq's `openai/gpt-oss-20b` model to generate an answer.

## 🚀 Features

- Upload HR Policy PDF
- Extract PDF text using PyMuPDF
- Clean and chunk the document
- Generate embeddings using Sentence Transformers
- Store embeddings in FAISS
- Perform semantic similarity search
- Retrieve relevant policy sections
- Generate answers using Groq
- Display retrieved policy sections
- Streamlit web interface

## 🧠 RAG Pipeline

```text
                HR Policy PDF
                      │
                      ▼
              PyMuPDF Extraction
                      │
                      ▼
                 Text Cleaning
                      │
                      ▼
                   Chunking
                      │
                      ▼
          Sentence Transformer Embeddings
                      │
                      ▼
                 FAISS Index
                      │
                      │
User Question ────────┤
                      ▼
             Question Embedding
                      │
                      ▼
             FAISS Similarity Search
                      │
                      ▼
             Relevant Policy Chunks
                      │
                      ▼
                  Groq LLM
              openai/gpt-oss-20b
                      │
                      ▼
                 Final Answer

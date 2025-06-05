# ragLearning

This project demonstrates a simple Retrieval-Augmented Generation (RAG) pipeline using LangChain and Hugging Face.

## 🔍 What It Does

- Loads a document (e.g., customer complaints)
- Splits it into chunks
- Converts chunks into embeddings
- Stores them in a FAISS vector store
- Uses a Hugging Face-hosted LLM (e.g., Flan-T5) to answer questions based on retrieved chunks

## 🧰 Technologies Used

- Python
- LangChain
- Hugging Face Hub
- FAISS
- Sentence Transformers

## 🚀 How to Run

1. Clone the repo
2. Install dependencies:
   ```bash
   pip install langchain langchain-community huggingface_hub faiss-cpu
   ```

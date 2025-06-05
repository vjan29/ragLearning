from langchain_community.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter

import os

# Step 1: Set your Hugging Face API token
os.environ["HUGGINGFACEHUB_API_TOKEN"] = "insert ur token here"  # Replace with your actual token

# Step 2: Load your document (e.g., customer complaints)
loader = TextLoader("complaints.txt")
documents = loader.load()

# Step 3: Split the text into chunks
text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
docs = text_splitter.split_documents(documents)

# Step 4: Create embeddings and vector store
embeddings = HuggingFaceEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)

# Step 5: Initialize the Hugging Face model
llm = HuggingFaceHub(
    repo_id="google/flan-t5-large",
    model_kwargs={"temperature": 0.5, "max_length": 512}
)

# Step 6: Create the RAG chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Step 7: Ask a question
query = "Summarize the main issues customers are complaining about."
response = qa_chain.run(query)

# Step 8: Print the result
print("Summary of complaints:")
print(response)

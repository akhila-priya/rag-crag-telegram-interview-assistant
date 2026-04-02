import os
from dotenv import load_dotenv

from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Step 1: Load DOCX
loader = Docx2txtLoader("data/RAG PDF.docx")
documents = loader.load()

# Step 2: Chunking
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Chunks ready:", len(chunks))

# Step 3: OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Step 4: Store in Pinecone
vector_store = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name="openai-rag-index"
)

print("Vectors stored successfully in Pinecone")
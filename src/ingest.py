from langchain_community.document_loaders import Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Step 1: Load DOCX
loader = Docx2txtLoader("data/RAG PDF.docx")
documents = loader.load()

print("Total documents loaded:", len(documents))

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print("Total chunks created:", len(chunks))

if len(chunks) > 0:
    print("\nFirst chunk preview:\n")
    print(chunks[0].page_content)
else:
    print("No chunks created")
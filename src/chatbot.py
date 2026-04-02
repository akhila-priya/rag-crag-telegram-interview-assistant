from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

load_dotenv()

# Step 1: connect embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Step 2: connect Pinecone
vector_store = PineconeVectorStore(
    index_name="openai-rag-index",
    embedding=embeddings
)

# Step 3: retrieve relevant chunks
query = "What is work performance and ownership?"

results = vector_store.similarity_search(query, k=3)

print("Top retrieved chunks:\n")

for i, doc in enumerate(results, 1):
    print(f"\nChunk {i}:\n")
    print(doc.page_content[:500])

# Step 4: generate final answer
context = "\n\n".join([doc.page_content for doc in results])

llm = ChatOpenAI(model="gpt-4o-mini")

prompt = f"""
Answer the question only using the context below.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\nFinal Answer:\n")
print(response.content)
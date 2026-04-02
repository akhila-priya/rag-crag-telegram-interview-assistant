import os
from dotenv import load_dotenv
from telegram import Update
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
)

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

# OpenAI Embeddings
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

# Pinecone Vector Store
vector_store = PineconeVectorStore(
    index_name="openai-rag-index",
    embedding=embeddings
)

# OpenAI Chat Model
llm = ChatOpenAI(
    model="gpt-4o-mini"
)

# Start command
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    context.user_data["history"] = []
    await update.message.reply_text(
        "Hi 👋 I am your RAG + CRAG Interview Assistant.\nAsk me any interview question."
    )

# Message handler with memory + corrective RAG
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.message.text

    # Load conversation history
    history = context.user_data.get("history", [])

    # Save user message
    history.append(f"User: {query}")

    # Keep last 4 messages
    history = history[-4:]

    history_text = "\n".join(history)

    # -------- CRAG LAYER --------
    scored_results = vector_store.similarity_search_with_score(query, k=3)

    correction_note = ""

    if scored_results:
        best_score = scored_results[0][1]

        # Low confidence → broader retrieval
        if best_score > 0.5:
            docs = vector_store.similarity_search(query, k=7)
            correction_note = "⚠ Low confidence detected. Using broader retrieval.\n\n"
        else:
            docs = [doc for doc, score in scored_results]
    else:
        docs = []

    # Fallback
    if not docs:
        await update.message.reply_text(
            "I could not find relevant information in the knowledge base."
        )
        return

    # Combine retrieved chunks
    context_text = "\n\n".join([doc.page_content for doc in docs])

    # Better prompt
    prompt = f"""
You are an expert interview preparation assistant.

Use BOTH:
1. previous conversation history
2. retrieved context

Conversation History:
{history_text}

Retrieved Context:
{context_text}

Current User Question:
{query}

Rules:
- continue previous topic
- do not change role/topic unless user explicitly asks
- if user asks "give answers too", continue previous role
- answer clearly in structured format
"""

    # Generate response
    response = llm.invoke(prompt)

    final_response = correction_note + response.content

    # Save assistant response in history
    history.append(f"Assistant: {response.content}")
    history = history[-4:]
    context.user_data["history"] = history

    # Send reply
    await update.message.reply_text(final_response[:4000])

# Build bot
app = ApplicationBuilder().token(TOKEN).build()

# Add handlers
app.add_handler(CommandHandler("start", start))
app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

print("RAG + CRAG Telegram bot is running...")

# Start polling
app.run_polling()
# Enterprise RAG Q&A System for Product & Solution Documentation

A production-grade **Retrieval-Augmented Generation (RAG)** system that answers questions about a company’s **products and solutions** by scraping, indexing, and querying live website content.

This system:

- 🕷️ **Scrapes** product/solution pages (e.g., `/products`, `/solutions`)
- 🧠 **Indexes** content using semantic embeddings (SentenceTransformer + FAISS)
- 💬 **Generates cited answers** with **source URLs**, **snippets**, and **detailed metrics**
- 🚀 Supports both **CLI** and **REST API** interfaces

Built with **async-first design**, **webhook-based async ingestion**, and **zero code duplication**.

> ℹ️ Validated on a live fintech website offering global payment solutions, but designed to work with any product-focused site.

## 📁 Structure

- [`part1-cli/`](part1-cli/) — CLI tools for ingestion and querying
- [`part2-api/`](part2-api/) — FastAPI service with webhook support

## 🛠️ Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Set your Google API key
echo GOOGLE_API_KEY=your_api_key_here > .env

# (Windows) Run setup.bat to install and configure
setup.bat
```

## 🌟 Key Features

- **Async scraping** with intelligent content extraction
- **Cited answers** with traceable sources
- **Cost-aware metrics**: tokens, latency, estimated LLM cost
- **Webhook pattern** for long-running ingestion
- **Concurrent batch querying**

## To setup the environment

```
setup.bat
```

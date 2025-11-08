# Part 1: CLI RAG System

Command-line tools to build and query a knowledge base from a company’s product and solution documentation.

## ✨ Capabilities (Validated on Real Data)

- Answers questions like:
  - _“What is [Product] and its key features?”_
  - _“How does [Solution] handle cross-border payments?”_
  - _“Which payment methods are supported?”_
- Sources answers from real product pages (e.g., BizPay, Wallet, Collections, Checkout)
- Cites specific use cases: e-commerce, payroll, gaming, dollar-based apps

## ▶️ Usage

## Go into the appropriate folder

```
cd part1_cli
```

### Ingest Website

```bash
python ingest.py --url https://example.com
```

### Ask a Question

```bash
python query.py --question "What is BizPay and its key features?"
```

### Batch Mode - sequential

```bash
python query.py --questions questions.txt
```

❗ Runs questions one after another. Total time ≈ sum of individual latencies.

### Batch Mode - Concurrent - (Recommonded)

```bash
python query.py --questions questions.txt --concurrent
```

✅ Runs all questions in parallel. Total time ≈ longest single query.

## 📊 Output Includes

- **Answer**: Generated using retrieved context
- **Sources**:
  - Title
  - URL
  - 300-character snippet
- **Metrics**:
  - Total latency
  - Retrieval/LLM time
  - Input/output tokens
  - Estimated cost
  - Documents retrieved vs. used

> 💡 The system intelligently extracts content from `<main>`, `<article>`, or high-value sections, ignoring navigation/footer noise.

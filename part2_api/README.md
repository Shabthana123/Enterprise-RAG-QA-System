# Part 2: FastAPI RAG Service

REST API that exposes the RAG system with **non-blocking ingestion** and **concurrent batch querying**.

## 🌐 Endpoints

| Endpoint           | Method | Behavior                                                                     |
| ------------------ | ------ | ---------------------------------------------------------------------------- |
| `/api/ingest`      | POST   | Starts async ingestion; returns immediately; sends metrics to `callback_url` |
| `/api/query`       | POST   | Returns answer + sources + metrics for one question                          |
| `/api/query/batch` | POST   | Processes questions concurrently; optional `callback_url` for async response |

## ▶️ Run Instructions

### Terminal 1: Webhook Receiver

### Go into the part2_api folder

```
cd part2_api
```

```bash
python webhook_receiver.py --port 8001
```

### Terminal 2: API Server (main folder)

```bash
uvicorn part2_api.api:app --port 8000
```

## 📤 Sample Answer (Based on Real Scraped Content)

```json
{
  "answer": "The platform supports over 300 local payment methods across 100+ countries, including stablecoins, cards, bank transfers, and e-wallets.",
  "sources": [
    {
      "id": 1,
      "title": "Collections – Accept Global Payments",
      "url": "https://example.com/products/collections",
      "snippet": "Accept global payments in 40+ currencies and stablecoins... with 300+ local payment methods."
    }
  ],
  "metrics": {
    "total_latency_s": 1.5,
    "input_tokens": 464,
    "output_tokens": 45,
    "estimated_cost_usd": 0.000065
  }
}
```

## 🔁 Webhook Payload

```
{
  "status": "success",
  "metrics": {
    "total_time_s": 8.4,
    "pages_scraped": 18,
    "pages_failed": 1,
    "total_chunks_created": 18,
    "total_tokens_processed": 603,
    "errors": ["Failed to fetch ... (status 404)"]
  }
}
```

> 🔁 **Reuses all logic from `part1-cli`** — no duplication.
> ✅ **Webhooks deliver structured JSON** with full ingestion metrics upon completion.

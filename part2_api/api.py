# api.py
import asyncio
import logging
from typing import List, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import httpx

# From Part 1
from part1_cli.ingest import run_ingest
from part1_cli.query import run_single_question, run_questions   


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Web-api")
app = FastAPI(title="Web RAG API", version="1.0")

# --- Models ---
class IngestRequest(BaseModel):
    urls: List[str]
    callback_url: str

class QueryRequest(BaseModel):
    question: str

class SourceItem(BaseModel):
    id: int
    title: str
    url: str
    snippet: str

class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[SourceItem]
    metrics: dict

class BatchQueryRequest(BaseModel):
    questions: List[str]
    callback_url: Optional[str] = None

class BatchQueryResponse(BaseModel):
    results: List[QueryResponse]
    aggregate_metrics: dict

# --- Webhook helper ---
async def send_webhook(url: str, payload: dict):
    async with httpx.AsyncClient(timeout=10.0) as client:
        try:
            logger.info(f"Sending webhook to {url}")
            resp = await client.post(url, json=payload)
            logger.info(f"Webhook sent: {resp.status_code}")
        except Exception as e:
            logger.error(f"Webhook failed for {url}: {e}")

# --- Background ingestion ---
async def background_ingest(urls: List[str], callback_url: str):
    try:
        base_url = urls[0].strip().rstrip("/") if urls else "https://www.transfi.com"
        metrics = await run_ingest(base_url)
        payload = {"status": "success", "metrics": metrics}
    except Exception as e:
        logger.exception("Ingestion crashed")
        payload = {"status": "error", "error": str(e)}
    await send_webhook(callback_url, payload)

# --- Background batch webhook sender ---
async def send_batch_webhook(results: List[QueryResponse], agg: dict, url: str):
    payload = {
        "results": [r.dict() for r in results],
        "aggregate_metrics": agg
    }
    await send_webhook(url, payload)
    
# --- Endpoints ---
@app.post("/api/ingest")
async def ingest_endpoint(req: IngestRequest, background_tasks: BackgroundTasks):
    background_tasks.add_task(background_ingest, req.urls, req.callback_url)
    return {"message": "Ingestion started"}

@app.post("/api/query", response_model=QueryResponse)
async def query_single(req: QueryRequest):
    try:
        result = await run_single_question(req.question)
        return QueryResponse(
            question=result["question"],
            answer=result["answer"],
            sources=[
                SourceItem(
                    id=s["id"],
                    title=s["title"],
                    url=s["url"],
                    snippet=s["snippet"]
                ) for s in result["sources"]
            ],
            metrics=result["metrics"]
        )
    except Exception as e:
        logger.exception("Query failed")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/query/batch") 
async def query_batch(req: BatchQueryRequest, background_tasks: BackgroundTasks):
    try:
        results = await run_questions(req.questions, concurrent=True)
        formatted_results = []
        for r in results:
            formatted_sources = [
                SourceItem(
                    id=s["id"],
                    title=s["title"],
                    url=s["url"],
                    snippet=s["snippet"]
                ) for s in r["sources"]
            ]
            formatted_results.append(
                QueryResponse(
                    question=r["question"],
                    answer=r["answer"],
                    sources=formatted_sources,
                    metrics=r["metrics"]
                )
            )

        agg = {
            "total_questions": len(results),
            "sum_input_tokens": sum(r["metrics"].get("input_tokens", 0) for r in results),
            "sum_output_tokens": sum(r["metrics"].get("output_tokens", 0) for r in results),
            "sum_estimated_cost": sum(r["metrics"].get("estimated_cost_usd", 0.0) for r in results)
        }

        # Optional: async webhook if callback_url is provided
        if req.callback_url:
            background_tasks.add_task(send_batch_webhook, formatted_results, agg, req.callback_url)
            return {"message": "Batch query started", "callback_url": req.callback_url}
        else:
            return BatchQueryResponse(results=formatted_results, aggregate_metrics=agg)

    except Exception as e:
        logger.exception("Batch query failed")
        raise HTTPException(status_code=500, detail=str(e))
    
    #  3 end points works well
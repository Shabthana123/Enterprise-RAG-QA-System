# query.py
"""
async query.py — Async-first concurrent query runner for RAG.

Usage:
  # Single question
  python query.py --question "What is BizPay?"

  # Multiple questions from a file (sequential)
  python query.py --questions questions.txt
  
  # Multiple questions from a file (concurrent)
  python query.py --questions questions.txt --concurrent

Notes:
- Supports my_retriever.retrieve_documents that can be async or blocking.
- Supports my_llm.generate_answer that can be async or blocking.
- Normalizes outputs and prints rich metrics per question + aggregate metrics.
"""

import argparse
import asyncio
import time
import inspect
import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple, Union
from threading import Lock
import os
import sys

# Allow CLI run like "python part1_cli/query.py ..."
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your helpers (they may be async or blocking)
from part1_cli.my_retriever import retrieve_documents
from part1_cli.my_llm import generate_answer

# ---------- CONFIG ----------
TOP_K = 5
# Pricing defaults (Gemini Flash-Lite-ish example: input $0.10 / 1M, output $0.40 / 1M)
COST_PER_1M_INPUT = 0.10
COST_PER_1M_OUTPUT = 0.40
# --------------------------------

# print lock to avoid interleaved stdout when running concurrently
print_lock = Lock()


def format_money(v: float) -> str:
    return f"${v:,.6f}"


def safe_extract_result(result: Union[Tuple[str, List[Dict[str, Any]]], Dict[str, Any]]) -> Dict[str, Any]:
    """
    Normalize generate_answer return to a dict:
    - If generate_answer returns (answer, used_docs) => convert.
    - If generate_answer returns dict => pass through.
    """
    if result is None:
        return {"answer": "No answer generated.", "used_docs": [], "token_usage": {"input": 0, "output": 0, "total": 0}}
    if isinstance(result, (tuple, list)):
        # (answer, used_docs)
        ans = result[0] if len(result) > 0 else "No answer generated."
        used = result[1] if len(result) > 1 else []
        return {"answer": ans, "used_docs": used, "token_usage": {"input": 0, "output": 0, "total": 0}}
    if isinstance(result, dict):
        # ensure keys exist and handle different naming conventions
        tu = result.get("token_usage", {}) or {}
        input_tokens = tu.get("input", tu.get("prompt_token_count", tu.get("prompt_tokens", 0)))
        output_tokens = tu.get("output", tu.get("output_token_count", tu.get("candidates_token_count", 0)))
        total_tokens = tu.get("total", tu.get("total_token_count", (input_tokens or 0) + (output_tokens or 0)))
        return {
            "answer": result.get("answer", "") or result.get("text", "") or "No answer generated.",
            "used_docs": result.get("used_docs", result.get("used_documents", [])) or [],
            "token_usage": {
                "input": int(input_tokens or 0),
                "output": int(output_tokens or 0),
                "total": int(total_tokens or (int(input_tokens or 0) + int(output_tokens or 0)))
            }
        }
    # fallback
    return {"answer": str(result), "used_docs": [], "token_usage": {"input": 0, "output": 0, "total": 0}}


async def run_single_question(question: str) -> Dict[str, Any]:
    """
    Async wrapper to retrieve docs and call LLM for a single question.
    Returns a structured dict with answer, sources, and metrics.
    """
    start_clock = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{start_clock}] Starting: {question}")

    q_start = time.time()
    metrics: Dict[str, Any] = {}

    # ----- Retrieval (support async or blocking retrieve_documents) -----
    t0 = time.time()
    try:
        if inspect.iscoroutinefunction(retrieve_documents):
            # async retriever returns (results, metrics)
            retrieved_res = await retrieve_documents(question, TOP_K)
        else:
            # blocking retriever — run in thread
            retrieved_res = await asyncio.to_thread(retrieve_documents, question, TOP_K)
        # accomodate both forms: (results, metrics) or just results
        if isinstance(retrieved_res, tuple) or isinstance(retrieved_res, list) and len(retrieved_res) == 2:
            retrieved, retrieval_metrics = retrieved_res
        else:
            # assume just list of docs
            retrieved = retrieved_res
            retrieval_metrics = {}
    except Exception as e:
        # fallback: try single return form, else empty
        try:
            if inspect.iscoroutinefunction(retrieve_documents):
                retrieved = await retrieve_documents(question, TOP_K)
            else:
                retrieved = await asyncio.to_thread(retrieve_documents, question, TOP_K)
            # if still returned tuple, unpack
            if isinstance(retrieved, (tuple, list)) and len(retrieved) == 2:
                retrieved, retrieval_metrics = retrieved
            else:
                retrieval_metrics = {}
        except Exception as e2:
            retrieved = []
            retrieval_metrics = {}
            print(f"[ERROR] Retrieval failed for question '{question}': {e} / {e2}")

    retrieval_time = time.time() - t0
    metrics["retrieval_time_s"] = round(float(retrieval_metrics.get("Retrieval Time (s)", retrieval_time)), 6)
    metrics["embedding_time_s"] = float(retrieval_metrics.get("Embedding Time (s)", retrieval_metrics.get("Embedding Time(s)", 0.0)))
    metrics["documents_retrieved"] = int(retrieval_metrics.get("Documents Retrieved", len(retrieved) if retrieved else 0))

    # ----- LLM generation (support async or blocking generate_answer) -----
    t1 = time.time()
    try:
        if inspect.iscoroutinefunction(generate_answer):
            llm_result = await generate_answer(question, retrieved)
        else:
            # run blocking LLM in a thread to keep event loop responsive
            llm_result = await asyncio.to_thread(generate_answer, question, retrieved)
    except Exception as e:
        # On failure, set a fallback message
        llm_result = {"answer": f"LLM generation error: {e}", "used_docs": [], "token_usage": {"input": 0, "output": 0, "total": 0}}
    llm_time = time.time() - t1
    metrics["llm_time_s"] = round(llm_time, 6)

    # standardize output
    result = safe_extract_result(llm_result)
    answer_text = result["answer"]
    used_docs = result["used_docs"] or []
    token_usage = result.get("token_usage", {"input": 0, "output": 0, "total": 0})

    # Metrics: token counts
    metrics["input_tokens"] = int(token_usage.get("input", 0))
    metrics["output_tokens"] = int(token_usage.get("output", 0))
    metrics["total_tokens"] = int(token_usage.get("total", metrics["input_tokens"] + metrics["output_tokens"]))

    # ----- Post-processing (snippets + formatting) -----
    # Use high-resolution timer for micro measurement (was previously always appearing as 0)
    t2 = time.perf_counter()
    sources = []
    for i, d in enumerate(used_docs, start=1):
        title = d.get("title") or d.get("url") or d.get("source") or "Unknown"
        url = d.get("source") or d.get("url") or "N/A"
        snippet = (d.get("snippet") or d.get("text") or "")[:300].replace("\n", " ").strip()
        sources.append({"id": i, "title": title, "url": url, "snippet": snippet})
    post_time = time.perf_counter() - t2
    # store post-processing in seconds with microsecond precision
    metrics["post_processing_time_s"] = round(post_time, 6)
    metrics["documents_used_in_answer"] = len(used_docs)

    # estimated cost using per-million prices (configurable)
    cost = (metrics["input_tokens"] / 1_000_000) * COST_PER_1M_INPUT + (metrics["output_tokens"] / 1_000_000) * COST_PER_1M_OUTPUT
    metrics["estimated_cost_usd"] = float(cost)

    metrics["total_latency_s"] = round(time.time() - q_start, 6)

    return {
        "question": question,
        "answer": answer_text,
        "sources": sources,
        "metrics": metrics,
        "retrieved_docs": retrieved  # raw retrieved (for debugging if needed)
    }


def print_result_block(res: Dict[str, Any]):
    """
    Print single result in the requested structured format.
    Use print_lock to avoid interleaving in concurrent runs.
    """
    with print_lock:
        q = res["question"]
        answer = res["answer"]
        sources = res["sources"]
        m = res["metrics"]

        print()
        print(f"Question: {q}")
        print(f"Answer: {answer}\n")
        print("Sources:")
        if sources:
            for s in sources:
                # keep numbering from 1..n
                print(f"  {s['id']}. {s['title']} - {s['url']}")
                print(f"     Snippet: \"{s['snippet']}\"")
        else:
            print("  None")
        print("\nMetrics:")
        # format metrics to match requested style keys
        print(f"  Total Latency: {m.get('total_latency_s', 0):.6f}s")
        print(f"  Retrieval Time: {m.get('retrieval_time_s', 0):.6f}s")
        print(f"  Embedding Time: {m.get('embedding_time_s', 0):.6f}s")
        print(f"  LLM Time: {m.get('llm_time_s', 0):.6f}s")
        print(f"  Post-processing Time: {m.get('post_processing_time_s', 0):.6f}s")
        print(f"  Documents Retrieved: {m.get('documents_retrieved', 0)}")
        print(f"  Documents Used in Answer: {m.get('documents_used_in_answer', 0)}")
        print(f"  Input Tokens: {m.get('input_tokens', 0):,}")
        print(f"  Output Tokens: {m.get('output_tokens', 0):,}")
        print(f"  Estimated Cost: {format_money(m.get('estimated_cost_usd', 0.0))}")
        print()


async def run_questions(questions: List[str], concurrent: bool = False) -> List[Dict[str, Any]]:
    """
    Execute questions concurrently or sequentially. Returns list of result dicts in input order.
    """
    if not questions:
        return []

    tasks = []
    if concurrent:
        for q in questions:
            tasks.append(asyncio.create_task(run_single_question(q)))
        # gather preserving order: asyncio.gather returns results in the order of the tasks
        results = await asyncio.gather(*tasks)
    else:
        results = []
        for q in questions:
            results.append(await run_single_question(q))
    return results


def print_usage_instructions():
    print("\nUsage:")
    print("  # Single question")
    print('  python query.py --question "What is BizPay?"')
    print("\n  # Multiple questions from file")
    print("  python query.py --questions questions.txt")
    print("\n  # Batch processing (concurrent)")
    print("  python query.py --questions questions.txt --concurrent\n")


async def main_async():
    parser = argparse.ArgumentParser(description="Async query runner (retrieval + LLM)")
    parser.add_argument("--question", type=str, help="Single question input")
    parser.add_argument("--questions", type=str, help="Path to file with multiple questions (one per line)")
    parser.add_argument("--concurrent", action="store_true", help="Run multiple questions concurrently")
    args = parser.parse_args()

    if args.question:
        questions = [args.question.strip()]
    elif args.questions:
        p = Path(args.questions)
        if not p.exists():
            print(f"[ERROR] Questions file not found: {args.questions}")
            return
        with p.open("r", encoding="utf-8") as fh:
            questions = [line.strip() for line in fh if line.strip()]
    else:
        print_usage_instructions()
        return

    start_all = time.time()
    results = await run_questions(questions, concurrent=args.concurrent)
    total_all = time.time() - start_all

    # Print each result block (thread-safe)
    for r in results:
        print_result_block(r)

    # Aggregate metrics (simple sums / averages)
    agg = {
        "total_questions": len(results),
        "total_time_s": round(total_all, 6),
        "sum_retrieval_s": sum(r["metrics"].get("retrieval_time_s", 0) for r in results),
        "sum_llm_s": sum(r["metrics"].get("llm_time_s", 0) for r in results),
        "sum_post_s": sum(r["metrics"].get("post_processing_time_s", 0) for r in results),
        "sum_input_tokens": sum(r["metrics"].get("input_tokens", 0) for r in results),
        "sum_output_tokens": sum(r["metrics"].get("output_tokens", 0) for r in results),
        "sum_estimated_cost": sum(r["metrics"].get("estimated_cost_usd", 0.0) for r in results)
    }

    with print_lock:
        print("=" * 60)
        print("=== Aggregate Run Metrics ===")
        print(f"Questions run: {agg['total_questions']}")
        print(f"Wall-clock Total Time: {agg['total_time_s']:.6f}s")
        print(f"Total Retrieval Time (sum): {agg['sum_retrieval_s']:.6f}s")
        print(f"Total LLM Time (sum): {agg['sum_llm_s']:.6f}s")
        print(f"Total Post-processing Time (sum): {agg['sum_post_s']:.6f}s")
        print(f"Total Input Tokens: {agg['sum_input_tokens']:,}")
        print(f"Total Output Tokens: {agg['sum_output_tokens']:,}")
        print(f"Estimated Total Cost: {format_money(agg['sum_estimated_cost'])}")
        print("=" * 60)
        print()


if __name__ == "__main__":
    asyncio.run(main_async())

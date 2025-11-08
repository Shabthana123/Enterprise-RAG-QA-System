# my_llm.py

import os
import google.generativeai as genai
from dotenv import load_dotenv
import asyncio
import traceback

# Load environment variables
load_dotenv()
if not os.getenv("GOOGLE_API_KEY"):
    raise EnvironmentError("GOOGLE_API_KEY not set in .env")

genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))


async def generate_answer(question, retrieved_docs):
    """
    Asynchronous LLM answer generation using Gemini 2.5 Flash-Lite.
    Returns structured data compatible with query.py metrics and citation display.
    """
    if not retrieved_docs:
        return {
            "answer": "No relevant information found.",
            "used_docs": [],
            "token_usage": {"input": 0, "output": 0, "total": 0}
        }

    # Build compact and meaningful context
    context_parts = []
    for doc in retrieved_docs:
        title = doc.get("title", "")
        snippet = doc.get("snippet", "") or doc.get("text", "")
        url = doc.get("url", "") or doc.get("source", "")
        context_parts.append(f"Title: {title}\nURL: {url}\nContent: {snippet}")

    context = "\n\n".join(context_parts[:10])  # limit to 10 chunks for token safety

    prompt = f"""
You are an AI assistant that answers **only using the provided context**.

Context:
{context}

Question: {question}

Answer clearly and concisely, citing relevant sources when possible.
"""

    model_name = "models/gemini-2.5-flash-lite"
    try:
        model = genai.GenerativeModel(model_name)
        response = await model.generate_content_async(prompt)
    except Exception as e:
        print(f"[Warning] Primary model failed ({model_name}): {e}")
        traceback.print_exc()
        print("[Info] Falling back to 'models/gemma-3-4b-it'...")
        model = genai.GenerativeModel("models/gemma-3-4b-it")
        response = await model.generate_content_async(prompt)

    # Extract answer safely
    answer = getattr(response, "text", None)
    if not answer:
        try:
            answer = response.candidates[0].content.parts[0].text.strip()
        except Exception:
            answer = "No answer generated."

    # Extract token usage metrics (Gemini API returns this inside usage_metadata)
    usage = getattr(response, "usage_metadata", None)
    if usage:
        input_tokens = getattr(usage, "prompt_token_count", 0)
        output_tokens = getattr(usage, "candidates_token_count", 0)
        total_tokens = getattr(usage, "total_token_count", input_tokens + output_tokens)
    else:
        input_tokens = output_tokens = total_tokens = 0

    # Include only the top 2 used documents (as representative sources)
    used_docs = []
    for doc in retrieved_docs[:2]:
        used_docs.append({
            "title": doc.get("title", "Unknown"),
            "url": doc.get("url", doc.get("source", "N/A")),
            "snippet": (doc.get("snippet") or doc.get("text", ""))[:300].strip()
        })

    return {
        "answer": answer.strip(),
        "used_docs": used_docs,
        "token_usage": {
            "input": input_tokens,
            "output": output_tokens,
            "total": total_tokens
        }
    }



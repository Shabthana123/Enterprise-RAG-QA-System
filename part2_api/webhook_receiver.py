# webhook_receiver.py
import argparse
from datetime import datetime
from fastapi import FastAPI, Request
import uvicorn
import json

app = FastAPI(title="Webhook Receiver")


@app.post("/webhook")
async def webhook_endpoint(request: Request):
    try:
        payload = await request.json()
    except Exception:
        # Fallback if body is not JSON
        body = await request.body()
        payload = {"raw_body": body.decode("utf-8", errors="replace")}

    timestamp = datetime.now().isoformat()
    print(f"\n[{timestamp}] Webhook Received:")
    # Pretty-print JSON with indentation
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return {"status": "ok"}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()
    print(f"📡 Webhook receiver running at http://localhost:{args.port}/webhook")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
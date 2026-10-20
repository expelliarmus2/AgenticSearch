"""DashScope OpenAI-compatible proxy.

Exposes POST /v1/chat/completions on localhost:8000 and forwards to
DashScope (https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions),
so repo scripts that hardcode a local vLLM endpoint (e.g. ircot.py ->
http://localhost:8000/v1/chat/completions) work without any code change.

Usage:
    export DASHSCOPE_API_KEY=sk-xxx   # optional, defaults to the key below
    python scripts/dashscope_proxy.py [--port 8000] [--model qwen3.8-max]
"""
import argparse
import asyncio
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

DASHSCOPE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"
API_KEY = os.environ.get("DASHSCOPE_API_KEY", "sk-8635104661ef4703aca16a2d612aa0ed")

app = FastAPI()


@app.post("/v1/chat/completions")
async def proxy_chat_completions(request: Request):
    body = await request.json()
    body["model"] = MODEL
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
    }
    async with httpx.AsyncClient(timeout=600) as client:
        for attempt in range(4):
            try:
                resp = await client.post(DASHSCOPE_URL, headers=headers, json=body)
            except httpx.HTTPError as e:
                if attempt == 3:
                    raise
                print(f"[proxy] DashScope connection error ({e}); retry {attempt + 1}/3...")
                await asyncio.sleep(2 * (attempt + 1))
                continue
            if resp.status_code < 500:
                break
            if attempt == 3:
                break
            print(f"[proxy] DashScope returned {resp.status_code}; retry {attempt + 1}/3...")
            await asyncio.sleep(2 * (attempt + 1))
    return JSONResponse(status_code=resp.status_code, content=resp.json())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--model", type=str, default="qwen3.8-max")
    args = parser.parse_args()
    MODEL = args.model
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

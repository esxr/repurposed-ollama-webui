import asyncio
import json
from datetime import datetime

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

app = FastAPI()
OLLAMA_SERVER_URL = "http://localhost:11434"

class RelayMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):

        other_server_url = f'{OLLAMA_SERVER_URL}{request.url.path}'

        body = b""
        async for chunk in request.stream():
            body += chunk

        async with httpx.AsyncClient() as client:
            req_data = {
                "method": request.method,
                "url": other_server_url,
                "headers": request.headers.raw,
                "params": request.query_params,
                "content": body
            }

            response = await client.request(**req_data)
            return Response(response.content, status_code=response.status_code, headers=dict(response.headers))

# Add the middleware to the FastAPI application
app.add_middleware(RelayMiddleware)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")
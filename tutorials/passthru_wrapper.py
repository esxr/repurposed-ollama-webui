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
        # Check if the request is for the /api/chat route
        if request.url.path == "/api/chat":
            # If so, just call the next item in the middleware stack
            return await call_next(request)

        # Otherwise, handle the request as before
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

@app.post("/api/chat")
async def chat(request: Request):

    def process(input_data):
        # Process the input_data and return a string
        # This is a placeholder for actual processing logic
        processed_string = "Processed: " + str(input_data)
        return processed_string

    async def generate_ndjson(model: str, msg: str):
        for word in msg.split():
            yield json.dumps({
                "model": model,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "message": {
                    "role": "assistant",
                    "content": word + " "
                },
                "done": False
            }) + "\n"
            await asyncio.sleep(0.1)
        yield json.dumps({
            "model": model,
            "created_at": datetime.utcnow().isoformat() + "Z",
            "message": {"role": "assistant", "content": "."},
            "done": True
        }) + "\n"

    # initialize input
    input_string = ""
    # Extract input from Ollama WebUI request (MIME type: text/event-stream)
    async for bytes in request.stream():
        if bytes:
            input_string = bytes.decode()
        else:
            continue
    # Process input_data
    input_data = json.loads(input_string)
    output = process(input_data)
    
    # Stream output in Ollama WebUI desired format (MIME type: application/x-ndjson)
    return StreamingResponse(generate_ndjson(model=input_data["model"], msg=output), media_type="application/x-ndjson")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5000, log_level="info")

import os
import json
from pathlib import Path
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from llm_agent import get_mcp_config, get_agent_app

load_dotenv()

app = FastAPI(title="Google Drive Agent")


class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default"


@app.get("/")
async def index():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    async def generate():
        try:
            mcp_config = get_mcp_config()
            mcp_client = MultiServerMCPClient(connections=mcp_config)
            agent = await get_agent_app(mcp_client)
            config = {"configurable": {"thread_id": body.session_id}}

            async for state in agent.astream(
                {"messages": [("user", body.prompt)]},
                config=config,
                stream_mode="values",
            ):
                messages = state.get("messages", [])
                if not messages:
                    continue

                last_msg = messages[-1]

                # Emit agent_active events for any tool calls in this state
                tool_calls = getattr(last_msg, "tool_calls", None)
                if tool_calls:
                    for tc in tool_calls:
                        yield f"data: {json.dumps({'type': 'agent_active', 'agent': tc.get('name', '')})}\n\n"

                content = last_msg.content

                # Resolve list content (Gemini sometimes returns blocks)
                if isinstance(content, list):
                    parts = []
                    for block in content:
                        if isinstance(block, str):
                            parts.append(block)
                        elif isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            if "grounding_metadata" in block or "groundingMetadata" in block:
                                parts.append("\n\n---\n**Sources:** Web Search active.")
                    content = "".join(parts)
                else:
                    content = str(content)

                if content.strip():
                    yield f"data: {json.dumps({'type': 'content', 'content': content})}\n\n"

        except BaseException as e:
            actual = e.exceptions[0] if hasattr(e, "exceptions") else e
            yield f"data: {json.dumps({'type': 'error', 'content': f'{type(actual).__name__}: {actual}'})}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

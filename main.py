import os
import json
from datetime import datetime, timezone
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_mcp_adapters.client import MultiServerMCPClient
from llm_agent import get_mcp_config, get_agent_app

load_dotenv()

app = FastAPI(title="Google Drive Agent")

# In-memory audit log store keyed by session_id
audit_store: dict[str, list] = {}


class ChatRequest(BaseModel):
    prompt: str
    session_id: str = "default"


@app.get("/")
async def index():
    return HTMLResponse(Path("static/index.html").read_text(encoding="utf-8"))


@app.get("/health")
async def health():
    return {"status": "ok"}


def _serialise_message(msg) -> dict:
    """Convert a LangChain message object into a plain dict for the audit log."""
    msg_type = type(msg).__name__
    content = msg.content
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") if isinstance(b, dict) else str(b) for b in content
        )

    entry = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "type": msg_type,
        "content": str(content),
    }

    tool_calls = getattr(msg, "tool_calls", None)
    if tool_calls:
        entry["tool_calls"] = [
            {"name": tc.get("name"), "args": tc.get("args")} for tc in tool_calls
        ]

    tool_name = getattr(msg, "name", None)
    if tool_name:
        entry["tool_name"] = tool_name

    return entry


@app.post("/chat/stream")
async def chat_stream(body: ChatRequest):
    async def generate():
        seen_ids: set = set()
        turn_log: list = []

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

                # Capture any messages we haven't logged yet.
                # Skip AIMessages with no tool_calls — those are the orchestrator
                # reasoning/synthesis steps which just duplicate the tool output.
                for msg in messages:
                    msg_id = id(msg)
                    if msg_id not in seen_ids:
                        seen_ids.add(msg_id)
                        is_ai = type(msg).__name__ == "AIMessage"
                        has_tool_calls = bool(getattr(msg, "tool_calls", None))
                        if not is_ai or has_tool_calls:
                            turn_log.append(_serialise_message(msg))

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
            actual = e
            while hasattr(actual, "exceptions") and actual.exceptions:
                actual = actual.exceptions[0]
            yield f"data: {json.dumps({'type': 'error', 'content': f'{type(actual).__name__}: {actual}'})}\n\n"

        finally:
            # Append this turn's log to the session store
            if turn_log:
                session_log = audit_store.setdefault(body.session_id, [])
                session_log.append({
                    "turn": len(session_log) + 1,
                    "prompt": body.prompt,
                    "steps": turn_log,
                })

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/audit/{session_id}")
async def download_audit(session_id: str):
    log = audit_store.get(session_id)
    if not log:
        raise HTTPException(status_code=404, detail="No audit log found for this session.")
    return JSONResponse(
        content={"session_id": session_id, "turns": log},
        headers={"Content-Disposition": f"attachment; filename=audit_{session_id[:8]}.json"},
    )

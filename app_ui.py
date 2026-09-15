import os
import json
import asyncio
import concurrent.futures
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
from langchain_mcp_adapters.client import MultiServerMCPClient
from llm_agent import get_mcp_config, get_agent_app

# =========================================================
# 1. BOOTSTRAP: HEALTH CHECKS
# =========================================================
st.set_page_config(page_title="G-Workspace Agent v2026", layout="wide")
st.title("Workspace & Search Agent")

with st.sidebar:
    st.header("System Health")
    if os.getenv("GOOGLE_API_KEY"):
        st.success("Gemini API Key found")
    else:
        st.error("Missing GOOGLE_API_KEY")

    if os.getenv("GCP_OAUTH_JSON_RAW"):
        st.success("OAuth App Credentials found")
    else:
        st.error("Missing GCP_OAUTH_JSON_RAW")

# =========================================================
# 2. AGENT EXECUTION
# =========================================================
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        if isinstance(message["content"], dict):
            st.markdown(message["content"].get("clean", ""))
        else:
            st.markdown(message["content"])

async def run_agent_workflow(user_prompt):
    mcp_config = get_mcp_config()
    mcp_client = MultiServerMCPClient(connections=mcp_config)

    try:
        agent = await get_agent_app(mcp_client)
        config = {"configurable": {"thread_id": "workspace_session_001"}}

        final_result = {"clean": "", "raw": ""}

        async for state in agent.astream(
            {"messages": [("user", user_prompt)]},
            config=config,
            stream_mode="values",
        ):
            messages = state.get("messages", [])
            if messages:
                last_msg = messages[-1]

                final_result["raw"] = (
                    json.dumps(last_msg.content, indent=2)
                    if not isinstance(last_msg.content, str)
                    else last_msg.content
                )

                if isinstance(last_msg.content, list):
                    parts = []
                    for block in last_msg.content:
                        if isinstance(block, dict):
                            if block.get("type") == "text":
                                parts.append(block.get("text", ""))
                            if "grounding_metadata" in block or "groundingMetadata" in block:
                                parts.append("\n\n---\n**Sources Found:** Web Search active.")
                    final_result["clean"] = "".join(parts)
                else:
                    final_result["clean"] = str(last_msg.content)

        return final_result

    except BaseException as e:
        # Unwrap anyio ExceptionGroup to surface the real sub-exception
        actual = e.exceptions[0] if hasattr(e, "exceptions") else e
        import traceback
        detail = "".join(traceback.format_exception(type(actual), actual, actual.__traceback__))
        return {"clean": f"Agent Error: {type(actual).__name__}: {actual}", "raw": detail}

# =========================================================
# 3. CHAT INPUT & LOOP
# =========================================================
if prompt := st.chat_input("Ask about your Drive or search the web..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        clean_tab, debug_tab = st.tabs(["Response", "Raw Metadata"])

        with st.spinner("Searching & Thinking..."):
            try:
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, run_agent_workflow(prompt))
                    result = future.result()

                with clean_tab:
                    st.markdown(result["clean"])
                with debug_tab:
                    st.code(result["raw"], language="json")

                st.session_state.messages.append({"role": "assistant", "content": result})
            except Exception as e:
                st.error(f"Execution Error: {e}")

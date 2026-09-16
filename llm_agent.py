import os
import re
import json
from pathlib import Path
from ddgs import DDGS
from mcp.types import PaginatedRequestParams
from google.oauth2.credentials import Credentials
from googleapiclient.discovery import build
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401 (re-exported)

# Shared orchestrator memory — persists conversation history across turns
memory = MemorySaver()

# --- Prompts ---

ORCHESTRATOR_PROMPT = """You are a Google Drive coordinator.

You have the following tools available:
- 'reader_agent' — finds and reads files in Google Drive
- 'editor_agent' — creates, updates, or appends to Google Drive files (Docs and Sheets)
- 'coder_agent'  — handles ALL web-related tasks: quick lookups return a plain answer; research/analysis tasks return a structured report

Rules:
- Delegate Drive reads to reader_agent and Drive writes to editor_agent.
- Use coder_agent for ANY question involving live data, web research, or report generation.
- If coder_agent returns plain text (no ===DOC=== delimiter), relay it directly to the user.
- If coder_agent returns delimited output, ALWAYS do both:
    1. Extract text between ===DOC=== and ===SHEET===, call editor_agent to create a Google Doc with that content. Use a clear title from the user's request.
    2. Extract text between ===SHEET=== and ===END===, call editor_agent to create a Google Sheet with that CSV data. Use the same title with " - Data" appended.
- Pass the user's full request as the query when calling an agent.
- If a task requires reading first then writing, call reader_agent first, then editor_agent.
- Be concise in your final response — for Drive tasks, tell the user what Doc and Sheet were created."""

READER_PROMPT = """You are a Google Drive reader specialist.

You have access to:
1. 'list_drive_files' — search for files in Google Drive by name.
2. 'read_drive_file' — read the full contents of a file. Pass the filename.

Rules:
- When searching Drive, NEVER use the 'fullText' operator. Search by name only.
- Return the file contents clearly and completely."""

CODER_PROMPT = """You are a research analyst and report writer.

You have access to:
1. 'web_search'      — search the web for live competitor data, market trends, news, or any research topic.
2. 'execute_python'  — run Python code to process, structure, and format the research into a polished report.

First, decide which mode applies:

MODE A — QUICK LOOKUP: The user wants a simple factual answer (current price, today's news, a quick fact).
- Use web_search once or twice.
- Return a short, plain conversational answer. No delimiters, no Drive output.

MODE B — RESEARCH REPORT: The user wants analysis, a comparison, or content saved to Google Drive.
- Use web_search multiple times to gather thorough information.
- Use execute_python to clean, organise, and structure the data.
- Return output in EXACTLY this format — no preamble, no commentary outside the delimiters:

===DOC===
[A polished written report: executive summary, key findings, and comparison tables as plain-text pipe tables. Professional and concise — no markdown.]
===SHEET===
[Title row then data rows as comma-separated values. Include all quantitative comparisons — pricing, scores, market share, features, etc.]
===END===

Rules for MODE B:
- The DOC section is plain text (it goes into Google Docs).
- The SHEET section must be valid CSV — quote any fields that contain commas.
- Always include at least one comparison table in the DOC and matching data in the SHEET."""

EDITOR_PROMPT = """You are a Google Drive editor specialist.

You have access to:
1. 'update_drive_file' — replace the entire content of a Google Doc.
2. 'append_to_drive_file' — add text to the end of a Google Doc.
3. 'create_drive_doc' — create a new Google Doc with a title and optional content.
4. 'create_drive_sheet' — create a new Google Sheets spreadsheet with a title and optional rows.
5. 'update_drive_sheet' — replace the data in an existing Google Sheet.

Rules:
- For full rewrites use 'update_drive_file'. For additions use 'append_to_drive_file'.
- Confirm what action was taken in your response."""

async def get_agent_app(mcp_client: MultiServerMCPClient):
    # ----------------------------------------------------------------
    # MCP tools (Drive search) — same sanitisation as before
    # ----------------------------------------------------------------
    mcp_tools = await mcp_client.get_tools()

    final_mcp_tools = []
    for t in mcp_tools:
        if t.name in ["search", "list_files", "list_drive_files"]:
            t.name = "list_drive_files"
            t.description = "Find files by name. Input MUST be a simple string (the filename)."

        def wrap_tool(tool_to_wrap):
            original_func = tool_to_wrap.func

            async def safe_func(**kwargs):
                for key, value in kwargs.items():
                    if isinstance(value, dict):
                        inner_val = next((v for v in value.values() if isinstance(v, str)), str(value))
                        kwargs[key] = inner_val
                    elif isinstance(value, list):
                        kwargs[key] = str(value).replace("[", "").replace("]", "").replace("'", "")
                for junk in ["orderBy", "order_by", "pageSize", "page_size"]:
                    kwargs.pop(junk, None)
                q_key = next((k for k in ["query", "q"] if k in kwargs), None)
                if q_key:
                    raw_val = str(kwargs[q_key])
                    match = re.search(r"'(.*?)'", raw_val)
                    keyword = match.group(1) if match else raw_val.strip()
                    clean_kw = keyword.replace("name", "").replace("contains", "").replace("fullText", "").strip(" '=")
                    kwargs[q_key] = f"name contains '{clean_kw}' and trashed = false"
                    kwargs["spaces"] = "drive"
                return await original_func(**kwargs)

            tool_to_wrap.func = safe_func
            return tool_to_wrap

        final_mcp_tools.append(wrap_tool(t))

    # ----------------------------------------------------------------
    # Shared helpers
    # ----------------------------------------------------------------
    def _clean_content(text: str) -> str:
        import csv, io
        try:
            rows = list(csv.reader(io.StringIO(text)))
        except Exception:
            return text
        rows = [r for r in rows if any(c.strip() for c in r)]
        if not rows:
            return text
        max_col = max(
            (i for row in rows for i, c in enumerate(row) if c.strip()),
            default=0,
        )
        trimmed = [row[:max_col + 1] for row in rows]
        out = io.StringIO()
        csv.writer(out).writerows(trimmed)
        return out.getvalue()

    def _get_write_creds():
        token_path = Path.home() / ".mcp-gdrive" / "write-tokens.json"
        write_token_json = os.getenv("GCP_WRITE_TOKEN_JSON_RAW")
        if write_token_json and not token_path.exists():
            token_path.write_text(write_token_json)
        if not token_path.exists():
            raise FileNotFoundError("Write credentials not found. Run 'py auth_write.py' first.")
        return Credentials.from_authorized_user_info(json.loads(token_path.read_text()))

    async def _find_doc_id(filename: str) -> str | None:
        async with mcp_client.session("google_workspace") as session:
            cursor = None
            for _ in range(20):
                params = PaginatedRequestParams(cursor=cursor) if cursor else None
                listing = await session.list_resources(params=params)
                for resource in listing.resources:
                    if filename.lower() in resource.name.lower():
                        return str(resource.uri).replace("gdrive:///", "")
                cursor = getattr(listing, "nextCursor", None)
                if not cursor:
                    break
        return None

    # ----------------------------------------------------------------
    # Read tools
    # ----------------------------------------------------------------
    async def _read_drive_file(filename: str) -> str:
        async with mcp_client.session("google_workspace") as session:
            cursor = None
            for _ in range(20):
                params = PaginatedRequestParams(cursor=cursor) if cursor else None
                listing = await session.list_resources(params=params)
                for resource in listing.resources:
                    if filename.lower() in resource.name.lower():
                        content_result = await session.read_resource(str(resource.uri))
                        if content_result.contents:
                            raw = getattr(content_result.contents[0], "text", str(content_result.contents[0]))
                            return _clean_content(raw)
                        return "File found but is empty or cannot be read as text."
                cursor = getattr(listing, "nextCursor", None)
                if not cursor:
                    break
        return f"No file matching '{filename}' was found in Google Drive."

    read_file_tool = StructuredTool.from_function(
        coroutine=_read_drive_file,
        name="read_drive_file",
        description="Read the full text content of a Google Drive file. Pass the filename (or part of it).",
    )

    # ----------------------------------------------------------------
    # Write tools
    # ----------------------------------------------------------------
    async def _update_drive_file(filename: str, new_content: str) -> str:
        doc_id = await _find_doc_id(filename)
        if not doc_id:
            return f"No file matching '{filename}' found in Google Drive."
        creds = _get_write_creds()
        docs = build("docs", "v1", credentials=creds)
        doc = docs.documents().get(documentId=doc_id).execute()
        end_index = doc["body"]["content"][-1]["endIndex"] - 1
        requests = []
        if end_index > 1:
            requests.append({"deleteContentRange": {"range": {"startIndex": 1, "endIndex": end_index}}})
        requests.append({"insertText": {"location": {"index": 1}, "text": new_content}})
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
        return f"'{filename}' updated successfully."

    async def _append_to_drive_file(filename: str, content: str) -> str:
        doc_id = await _find_doc_id(filename)
        if not doc_id:
            return f"No file matching '{filename}' found in Google Drive."
        creds = _get_write_creds()
        docs = build("docs", "v1", credentials=creds)
        doc = docs.documents().get(documentId=doc_id).execute()
        end_index = doc["body"]["content"][-1]["endIndex"] - 1
        requests = [{"insertText": {"location": {"index": end_index}, "text": "\n" + content}}]
        docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
        return f"Content appended to '{filename}' successfully."

    async def _create_drive_doc(title: str, content: str) -> str:
        creds = _get_write_creds()
        docs = build("docs", "v1", credentials=creds)
        doc = docs.documents().create(body={"title": title}).execute()
        doc_id = doc["documentId"]
        if content:
            requests = [{"insertText": {"location": {"index": 1}, "text": content}}]
            docs.documents().batchUpdate(documentId=doc_id, body={"requests": requests}).execute()
        return f"Created new Google Doc '{title}' (ID: {doc_id})."

    async def _create_drive_sheet(title: str, rows: str) -> str:
        creds = _get_write_creds()
        sheets = build("sheets", "v4", credentials=creds)
        spreadsheet = sheets.spreadsheets().create(body={"properties": {"title": title}}).execute()
        sheet_id = spreadsheet["spreadsheetId"]
        if rows.strip():
            values = [r.split(",") for r in rows.strip().splitlines()]
            sheets.spreadsheets().values().update(
                spreadsheetId=sheet_id, range="A1",
                valueInputOption="USER_ENTERED", body={"values": values},
            ).execute()
        return f"Created Google Sheet '{title}' (ID: {sheet_id})."

    async def _update_drive_sheet(filename: str, rows: str) -> str:
        doc_id = await _find_doc_id(filename)
        if not doc_id:
            return f"No file matching '{filename}' found in Google Drive."
        creds = _get_write_creds()
        sheets = build("sheets", "v4", credentials=creds)
        sheets.spreadsheets().values().clear(spreadsheetId=doc_id, range="A1:ZZ10000", body={}).execute()
        if rows.strip():
            values = [r.split(",") for r in rows.strip().splitlines()]
            sheets.spreadsheets().values().update(
                spreadsheetId=doc_id, range="A1",
                valueInputOption="USER_ENTERED", body={"values": values},
            ).execute()
        return f"'{filename}' sheet data updated."

    update_tool = StructuredTool.from_function(
        coroutine=_update_drive_file,
        name="update_drive_file",
        description="Replace the full content of a Google Doc. Pass the filename and the complete new content.",
    )
    append_tool = StructuredTool.from_function(
        coroutine=_append_to_drive_file,
        name="append_to_drive_file",
        description="Append text to the end of an existing Google Doc. Pass the filename and the text to add.",
    )
    create_tool = StructuredTool.from_function(
        coroutine=_create_drive_doc,
        name="create_drive_doc",
        description="Create a new Google Doc with a given title and optional initial content.",
    )
    create_sheet_tool = StructuredTool.from_function(
        coroutine=_create_drive_sheet,
        name="create_drive_sheet",
        description=(
            "Create a new Google Sheets spreadsheet. Pass the title and optional rows as "
            "newline-separated, comma-separated values. Example rows: 'Name,Age\\nAlice,30\\nBob,25'"
        ),
    )
    update_sheet_tool = StructuredTool.from_function(
        coroutine=_update_drive_sheet,
        name="update_drive_sheet",
        description="Replace the data in an existing Google Sheet. Pass the filename and rows as newline-separated, comma-separated values.",
    )

    # ----------------------------------------------------------------
    # Model
    # ----------------------------------------------------------------
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0,
    )

    # ----------------------------------------------------------------
    # Sub-agent helper — lightweight tool-use loop.
    # Avoids nested LangGraph agents (which use asyncio.create_task
    # internally and break MCP contextvar-based session propagation).
    # ----------------------------------------------------------------
    async def _run_tool_loop(tools: list, system_prompt: str, query: str) -> str:
        from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage

        tool_map = {t.name: t for t in tools}
        bound = model.bind_tools(tools)
        messages = [SystemMessage(content=system_prompt), HumanMessage(content=query)]

        for _ in range(10):
            response = await bound.ainvoke(messages)
            messages.append(response)

            if not response.tool_calls:
                return response.content if isinstance(response.content, str) else str(response.content)

            for tc in response.tool_calls:
                tool = tool_map.get(tc["name"])
                result = await tool.ainvoke(tc["args"]) if tool else f"Unknown tool: {tc['name']}"
                messages.append(ToolMessage(content=str(result), tool_call_id=tc["id"]))

        return str(messages[-1].content)

    # ----------------------------------------------------------------
    # Python execution tool
    # ----------------------------------------------------------------
    def _execute_python(code: str) -> str:
        import subprocess, sys, tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
            f.write(code)
            tmp_path = f.name
        try:
            result = subprocess.run(
                [sys.executable, tmp_path],
                capture_output=True, text=True, timeout=30,
            )
            output = result.stdout
            if result.stderr:
                output += f"\nSTDERR:\n{result.stderr}"
            return output.strip() or "Code executed successfully (no output)."
        except subprocess.TimeoutExpired:
            return "Error: code execution timed out after 30 seconds."
        finally:
            os.unlink(tmp_path)

    execute_python_tool = StructuredTool.from_function(
        func=_execute_python,
        name="execute_python",
        description="Execute a Python code snippet and return stdout/stderr. Use this to process, structure, and format research data into a report.",
    )

    # ----------------------------------------------------------------
    # Web search tool
    # ----------------------------------------------------------------
    def _web_search(query: str) -> str:
        results = list(DDGS().text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"**{r['title']}**\n{r['href']}\n{r['body']}" for r in results
        )

    web_search_tool = StructuredTool.from_function(
        func=_web_search,
        name="web_search",
        description="Search the web for live information or current events. Pass a search query string.",
    )

    # ----------------------------------------------------------------
    # Sub-agents wrapped as orchestrator tools
    # ----------------------------------------------------------------
    reader_tools = final_mcp_tools + [read_file_tool]
    editor_tools = [update_tool, append_tool, create_tool, create_sheet_tool, update_sheet_tool]
    coder_tools = [web_search_tool, execute_python_tool]

    async def call_reader(query: str) -> str:
        return await _run_tool_loop(reader_tools, READER_PROMPT, query)

    async def call_editor(query: str) -> str:
        return await _run_tool_loop(editor_tools, EDITOR_PROMPT, query)

    async def call_coder(query: str) -> str:
        return await _run_tool_loop(coder_tools, CODER_PROMPT, query)

    reader_tool = StructuredTool.from_function(
        coroutine=call_reader,
        name="reader_agent",
        description="Find and read Google Drive files. Pass the user's full request as the query.",
    )
    editor_tool = StructuredTool.from_function(
        coroutine=call_editor,
        name="editor_agent",
        description="Create, update, or append to Google Drive files. Pass the user's full request as the query.",
    )
    coder_tool = StructuredTool.from_function(
        coroutine=call_coder,
        name="coder_agent",
        description="Research a topic on the web, process the findings with Python, and return a formatted report ready to save. Pass the user's full request as the query.",
    )

    # ----------------------------------------------------------------
    # Orchestrator — the only agent main.py talks to
    # ----------------------------------------------------------------
    return create_agent(
        model=model,
        tools=[reader_tool, editor_tool, coder_tool],
        checkpointer=memory,
        system_prompt=ORCHESTRATOR_PROMPT,
    )

def get_mcp_config():
    """Returns MCP connection config after writing credential files to disk."""
    mcp_dir = os.path.join(os.path.expanduser("~"), ".mcp-gdrive")
    os.makedirs(mcp_dir, exist_ok=True)
    creds_path = os.path.join(mcp_dir, "gcp-oauth.keys.json")
    token_path = os.path.join(mcp_dir, "tokens.json")

    oauth_json = os.getenv("GCP_OAUTH_JSON_RAW")
    token_json = os.getenv("GCP_TOKEN_JSON_RAW")

    if oauth_json and not os.path.exists(creds_path):
        with open(creds_path, "w") as f:
            f.write(oauth_json)
    if token_json and not os.path.exists(token_path):
        with open(token_path, "w") as f:
            f.write(token_json)

    return {
        "google_workspace": {
            "transport": "stdio",
            "command": "mcp-server-gdrive",
            "args": [],
            "env": {
                "GDRIVE_OAUTH_PATH": creds_path,
                "GDRIVE_CREDENTIALS_PATH": token_path,
            },
        }
    }
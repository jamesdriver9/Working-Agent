import os
import re
import json
import time
from mcp.types import PaginatedRequestParams
from pathlib import Path
from ddgs import DDGS
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
- Be concise in your final response — for Drive tasks, tell the user what Doc and Sheet were created.

Multi-step workflow — use this automatically when the request implies it:

RESEARCH REPORT (no existing file):
When the user wants a new report, analysis, or brief on a topic:
  Step 1 — Call coder_agent with the full research request.
  Step 2 — Call editor_agent to save the result as a new Google Doc (and Sheet if data is present).

CONTEXTUAL REPORT (enrich an existing Drive file):
When the user wants to update, review, or enrich an existing Drive document with live research:
  Step 1 — Call reader_agent to retrieve the relevant Drive file.
  Step 2 — If reader_agent returns file content, call coder_agent with:
            "EXISTING CONTEXT:\\n<reader output>\\n\\nRESEARCH REQUEST:\\n<what to research and produce>"
            If reader_agent returns an error or says the file was not found, proceed WITHOUT the file:
            call coder_agent with just the research request and note the file could not be retrieved.
  Step 3 — Call editor_agent to save the result as a new Google Doc (and Sheet if data is present).

Always infer which pattern applies from natural language. Never stop because reader_agent failed — always proceed to coder_agent."""

READER_PROMPT = """You are a Google Drive reader specialist.

You have access to:
1. 'list_drive_files' — search for files in Google Drive by name.
2. 'read_drive_file' — read the full contents of a file. Pass the filename or part of it.

Rules:
- When searching Drive, NEVER use the 'fullText' operator. Search by name only.
- Always call 'list_drive_files' first to search for the file.
- If 'list_drive_files' returns no results or you are unsure, STILL call 'read_drive_file' directly with the filename — it searches all drives including Shared Drives.
- Return the file contents clearly and completely."""

CODER_PROMPT = """You are a research analyst and report writer.

You have access to:
1. 'web_search'      — search the web for live competitor data, market trends, news, or any research topic.
2. 'execute_python'  — run Python code to process, structure, and format the research into a polished report.

First, decide which mode applies:

MODE A — QUICK LOOKUP: The user wants a simple factual answer (current price, today's news, a quick fact).
- Use web_search once or twice.
- Return a short, plain conversational answer. No delimiters, no Drive output.
- Always end with a Sources section listing the page title and URL for every result you used.

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
- Always end the DOC section with a Sources section listing the page title and URL for every result used.
- The SHEET section must be valid CSV — quote any fields that contain commas.
- Always include at least one comparison table in the DOC and matching data in the SHEET.

MODE C — CONTEXTUAL REPORT: The query contains "EXISTING CONTEXT:" followed by a Drive document, and "RESEARCH REQUEST:" with what to produce.
- Read the existing context carefully — it is the user's own notes, agenda, or brief.
- Use web_search to find current intelligence that supplements and enriches that context (news, financials, competitor data, industry trends — whatever is most relevant).
- Use execute_python to merge and structure both sources into a polished output.
- Return output in the ===DOC=== / ===SHEET=== / ===END=== format.
- The DOC must have clear sections: first a summary of the existing context, then a Research & Intelligence section with findings, then a combined Recommendations or Key Talking Points section.
- Always end with a Sources section listing URLs for every web result used."""

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
            # Capture whichever callable the tool actually uses at runtime.
            # Async MCP tools from langchain-mcp-adapters use .coroutine, not .func,
            # so we must wrap .coroutine or the sanitisation is silently bypassed.
            original_coroutine = tool_to_wrap.coroutine
            original_func = tool_to_wrap.func

            async def safe_func(**kwargs):
                for key, value in list(kwargs.items()):
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
                if original_coroutine:
                    return await original_coroutine(**kwargs)
                return original_func(**kwargs)

            tool_to_wrap.coroutine = safe_func
            tool_to_wrap.func = None
            return tool_to_wrap

        final_mcp_tools.append(wrap_tool(t))

    # ----------------------------------------------------------------
    # Shared helpers
    # ----------------------------------------------------------------
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
    def _get_read_creds():
        mcp_dir = Path.home() / ".mcp-gdrive"
        token_path = mcp_dir / "tokens.json"
        oauth_path = mcp_dir / "gcp-oauth.keys.json"

        read_token_json = os.getenv("GCP_TOKEN_JSON_RAW")
        if read_token_json and not token_path.exists():
            token_path.write_text(read_token_json)
        if not token_path.exists():
            raise FileNotFoundError("Read credentials not found. Ensure GCP_TOKEN_JSON_RAW is set or run the auth flow.")

        tokens = json.loads(token_path.read_text())

        # tokens.json already has client_id/client_secret (write-token format) — use directly
        if "client_id" in tokens and "client_secret" in tokens:
            return Credentials.from_authorized_user_info(tokens)

        # MCP server stores Node.js OAuth tokens without client credentials.
        # Combine with gcp-oauth.keys.json to build a valid Credentials object.
        if not oauth_path.exists():
            raise FileNotFoundError("gcp-oauth.keys.json not found — cannot construct read credentials.")
        oauth_raw = json.loads(oauth_path.read_text())
        client_info = oauth_raw.get("web", oauth_raw.get("installed", oauth_raw))
        combined = {
            "client_id": client_info["client_id"],
            "client_secret": client_info["client_secret"],
            "refresh_token": tokens.get("refresh_token", ""),
            "token_uri": "https://oauth2.googleapis.com/token",
        }
        return Credentials.from_authorized_user_info(combined)

    async def _read_drive_file(filename: str) -> str:
        import asyncio

        def _sync_read():
            try:
                creds = _get_read_creds()
            except Exception as e:
                return f"Credential error: {e}"

            try:
                drive = build("drive", "v3", credentials=creds)
                results = drive.files().list(
                    q=f"name contains '{filename}' and trashed = false",
                    fields="files(id, name, mimeType)",
                    pageSize=10,
                    includeItemsFromAllDrives=True,
                    supportsAllDrives=True,
                    corpora="allDrives",
                ).execute()
            except Exception as e:
                return f"Drive search error: {e}"

            files = results.get("files", [])
            if not files:
                return f"No file matching '{filename}' was found in Google Drive."

            file = files[0]
            file_id = file["id"]
            mime_type = file.get("mimeType", "")
            name = file.get("name", filename)

            try:
                if "google-apps.document" in mime_type:
                    content = drive.files().export(fileId=file_id, mimeType="text/plain").execute()
                elif "google-apps.spreadsheet" in mime_type:
                    content = drive.files().export(fileId=file_id, mimeType="text/csv").execute()
                else:
                    content = drive.files().get_media(fileId=file_id).execute()
                return content.decode("utf-8") if isinstance(content, bytes) else str(content)
            except Exception as e:
                return f"Found '{name}' but could not read its content: {e}"

        return await asyncio.to_thread(_sync_read)

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
                if tool:
                    try:
                        result = await tool.ainvoke(tc["args"])
                    except Exception as err:
                        inner = err
                        while hasattr(inner, "exceptions") and inner.exceptions:
                            inner = inner.exceptions[0]
                        result = f"Tool error ({tc['name']}): {type(inner).__name__}: {inner}"
                else:
                    result = f"Unknown tool: {tc['name']}"
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
    def _web_search(query: str, timelimit: str = "m") -> str:
        """timelimit: 'd' (day), 'w' (week), 'm' (month), 'y' (year), or None for all time."""
        results = []
        last_error = None
        for attempt in range(3):
            try:
                results = list(DDGS().text(query, max_results=5, timelimit=timelimit))
                if results:
                    break
            except Exception as e:
                last_error = e
                if attempt < 2:
                    time.sleep(2 ** attempt)

        # Fallback: retry without time filter if timelimited search returned nothing
        if not results and timelimit:
            try:
                results = list(DDGS().text(query, max_results=5))
            except Exception as e:
                last_error = e

        if not results:
            if last_error:
                return f"Web search failed: {type(last_error).__name__}: {last_error}"
            return "No results found for this query."
        return "\n\n".join(
            f"**{r['title']}**\n{r['href']}\n{r['body']}" for r in results
        )

    web_search_tool = StructuredTool.from_function(
        func=_web_search,
        name="web_search",
        description=(
            "Search the web for live information or current events. "
            "Pass a search query string. "
            "Optionally pass timelimit='d' (past day), 'w' (past week), "
            "'m' (past month), or 'y' (past year) to filter for recent results."
        ),
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
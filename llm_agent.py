import os
import re
from ddgs import DDGS
from mcp.types import PaginatedRequestParams
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import StructuredTool
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401 (re-exported)

# Initialize memory checkpointer
memory = MemorySaver()

# System Prompt
GDRIVE_INTEL_PROMPT = """You are a Google Drive Intelligence Agent.

You have three tools available:
1. 'list_drive_files' — search for files in Google Drive by name. Use this when the user wants to know what files exist.
2. 'read_drive_file' — read the full contents of a file. Pass the filename (or part of it). Use this whenever the user wants to see what is inside a file.
3. 'web_search' — for live internet queries: news, facts, prices, current events. Pass a plain search query string.

Rules:
- If the user asks about file contents, always call 'read_drive_file' directly with the filename. Do not call list_drive_files first.
- Use 'web_search' for anything requiring live data — do not answer current events from memory.
- When searching Drive, NEVER use the 'fullText' operator. Search by name only.
- Be concise and helpful."""

async def get_agent_app(mcp_client: MultiServerMCPClient):
    mcp_tools = await mcp_client.get_tools()
    
    final_mcp_tools = []
    for t in mcp_tools:
        # Standardize the search tool name
        if t.name in ["search", "list_files", "list_drive_files"]:
            t.name = "list_drive_files"
            t.description = "Find files by name. Input MUST be a simple string (the filename)."
        
        def wrap_tool(tool_to_wrap):
            original_func = tool_to_wrap.func
            
            async def safe_func(**kwargs):
                # 1. 🚨 THE "DEEP FLATTEN": If Gemini sends a dict, rip the string out
                # This fixes the "Invalid Value" error caused by complex tool inputs
                for key, value in kwargs.items():
                    if isinstance(value, dict):
                        # Take the first string value inside the dict, or just stringify the whole thing
                        inner_val = next((v for v in value.values() if isinstance(v, str)), str(value))
                        kwargs[key] = inner_val
                    elif isinstance(value, list):
                        kwargs[key] = str(value).replace("[", "").replace("]", "").replace("'", "")

                # 2. 🚨 THE UNIVERSAL STRIPPER: Remove sorting from EVERY tool
                for junk in ["orderBy", "order_by", "pageSize", "page_size"]:
                    kwargs.pop(junk, None)
                
                # 3. 🚨 THE QUERY RE-WRITER: Prevent FullText/Sorting crashes
                q_key = next((k for k in ["query", "q"] if k in kwargs), None)
                if q_key:
                    raw_val = str(kwargs[q_key])
                    # Extract keyword from quotes if present
                    match = re.search(r"'(.*?)'", raw_val)
                    keyword = match.group(1) if match else raw_val.strip()
                    # Kill operators that trigger "FullText" mode
                    clean_kw = keyword.replace("name", "").replace("contains", "").replace("fullText", "").strip(" '=")
                    # Force clean name search
                    kwargs[q_key] = f"name contains '{clean_kw}' and trashed = false"
                    kwargs["spaces"] = "drive"

                return await original_func(**kwargs)
            
            tool_to_wrap.func = safe_func
            return tool_to_wrap
            
        final_mcp_tools.append(wrap_tool(t))

    # Tool that pages through Drive resources to find a file by name, then reads it
    async def _read_drive_file(filename: str) -> str:
        async with mcp_client.session("google_workspace") as session:
            cursor = None
            for _ in range(20):  # cap at 20 pages (200 files)
                params = PaginatedRequestParams(cursor=cursor) if cursor else None
                listing = await session.list_resources(params=params)

                for resource in listing.resources:
                    if filename.lower() in resource.name.lower():
                        content_result = await session.read_resource(str(resource.uri))
                        if content_result.contents:
                            return getattr(content_result.contents[0], "text", str(content_result.contents[0]))
                        return "File found but is empty or cannot be read as text."

                cursor = getattr(listing, "nextCursor", None)
                if not cursor:
                    break

            return f"No file matching '{filename}' was found in Google Drive."

    read_file_tool = StructuredTool.from_function(
        coroutine=_read_drive_file,
        name="read_drive_file",
        description="Read the full text content of a Google Drive file. Pass the filename (or part of it) and the tool will find and return the contents automatically.",
    )
    final_mcp_tools.append(read_file_tool)

    # Initialize Model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0 # Stable for tool usage
    )
    
    async def _web_search(query: str) -> str:
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=5))
        if not results:
            return "No results found."
        return "\n\n".join(
            f"**{r['title']}**\n{r['body']}\nSource: {r['href']}"
            for r in results
        )

    web_search_tool = StructuredTool.from_function(
        coroutine=_web_search,
        name="web_search",
        description="Search the web for current news, facts, or any live information. Input is a plain search query string.",
    )

    return create_agent(
        model=model,
        tools=final_mcp_tools + [web_search_tool],
        checkpointer=memory,
        system_prompt=GDRIVE_INTEL_PROMPT
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
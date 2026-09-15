import os
import re
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import create_agent
from langchain_mcp_adapters.client import MultiServerMCPClient  # noqa: F401 (re-exported)

# Initialize memory checkpointer
memory = MemorySaver()

# System Prompt
GDRIVE_INTEL_PROMPT = """You are a Google Drive Intelligence Agent.
Today is March 28, 2026. 

You have two modes of operation:
1. PRIVATE DRIVE FILES: If the user asks about their personal files, documents, or spreadsheets, you MUST use the 'list_drive_files' tool.
2. THE WEB (Recipes, News, Facts): If the user asks for general web information, DO NOT use any tools. Simply answer the question directly. Your direct answers are natively connected to Google Search.

Rules:
- If you search Drive and find nothing, do not apologize. Just answer directly to trigger your native web knowledge.
- Be concise and helpful.
- When searching Drive, NEVER use the 'fullText' operator. Search by name instead."""

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

    # ... remaining model and create_agent code ...

    # Initialize Model
    model = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=os.getenv("GOOGLE_API_KEY"),
        temperature=0.0 # Stable for tool usage
    )
    
    # Bind tools including native web grounding
    model_with_tools = model.bind_tools(final_mcp_tools + [{"google_search": {}}])

    return create_agent(
        model=model_with_tools,
        tools=final_mcp_tools,
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
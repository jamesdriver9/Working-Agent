"""
Run this once to authorize write access to Google Drive & Docs.
It opens a browser for Google sign-in, then saves the token to
~/.mcp-gdrive/write-tokens.json

Usage:
    py auth_write.py
"""
import json
import os
from pathlib import Path
from dotenv import load_dotenv
from google_auth_oauthlib.flow import InstalledAppFlow

SCOPES = [
    "https://www.googleapis.com/auth/drive",
    "https://www.googleapis.com/auth/documents",
    "https://www.googleapis.com/auth/spreadsheets",
]

load_dotenv()

mcp_dir = Path.home() / ".mcp-gdrive"
mcp_dir.mkdir(exist_ok=True)
client_secrets_path = mcp_dir / "gcp-oauth.keys.json"
write_token_path = mcp_dir / "write-tokens.json"

# Write the client secrets from .env if not already on disk
oauth_json = os.getenv("GCP_OAUTH_JSON_RAW")
if not oauth_json:
    raise SystemExit("GCP_OAUTH_JSON_RAW is not set in your .env file.")
client_secrets_path.write_text(oauth_json)

flow = InstalledAppFlow.from_client_secrets_file(str(client_secrets_path), SCOPES)
creds = flow.run_local_server(port=0)

write_token_path.write_text(creds.to_json())
print(f"\nWrite token saved to: {write_token_path}")
print("Paste the contents of that file into GCP_WRITE_TOKEN_JSON_RAW in your .env")

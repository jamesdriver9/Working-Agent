# 1. Using the March 2026 Stable Debian 13 (Trixie) base
# Debian 13.4 was released March 14, 2026, making it the most secure current base.
FROM python:3.12-slim-trixie

# 2. Set Environment Variables for Production
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_TELEMETRY_ENABLED=false

# 3. Install System Dependencies + Node.js 22 LTS
# libmagic1 is required for Drive file type identification
# poppler-utils is added for PDF text extraction support
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    ca-certificates \
    gnupg \
    libmagic1 \
    poppler-utils \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y nodejs \
    && npm install -g npm@latest \
    && rm -rf /var/lib/apt/lists/*

# 4. Security: Non-root user with UID 1001 for Azure Managed Identity compliance
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# 5. Install Python Dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Pre-fetch the Unified Google Workspace MCP Server
# We pre-install this so the first user request in Azure doesn't have to wait for a download.
RUN npm install -g @piotr-agier/google-drive-mcp

# 7. Copy Application Code & Set Permissions
COPY --chown=appuser:appuser . .

# 8. Set up Workspace & Token Storage
# We use /app/credentials to store the gcp-oauth.keys.json and tokens.json
# Note: In Azure, we will pass these as Env Vars (JSON strings), but the MCP 
# server still needs a writable directory for its session cache.
RUN mkdir -p /app/credentials /tmp/mcp-cache && \
    chown -R appuser:appuser /app /home/appuser /tmp/mcp-cache

USER appuser

# 9. Expose Streamlit Port
EXPOSE 8501

# 10. Azure-Optimized Healthcheck
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8501/_stcore/health || exit 1

# 11. Start Command
# Using --server.enableXsrfProtection=false for better compatibility with Azure Front Door
ENTRYPOINT ["streamlit", "run", "app_ui.py"]
CMD [ \
    "--server.port=8501", \
    "--server.address=0.0.0.0", \
    "--server.enableCORS=false", \
    "--server.enableXsrfProtection=false", \
    "--server.fileWatcherType=none" \
    ]
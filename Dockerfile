# Debian 13 (Trixie) slim base
FROM python:3.12-slim-trixie

# Environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

# System dependencies + Node.js 22 LTS (required for MCP server npm package)
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

# Non-root user — UID 1001 for Azure Managed Identity compliance
RUN groupadd -g 1001 appuser && \
    useradd -u 1001 -r -g appuser -m -d /home/appuser appuser

WORKDIR /app

# Python dependencies
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Google Drive MCP server (pre-installed so first request doesn't wait)
RUN npm install -g @piotr-agier/google-drive-mcp

# Application code
COPY --chown=appuser:appuser . .

# Writable directories for credentials and MCP session cache
RUN mkdir -p /app/credentials /tmp/mcp-cache && \
    chown -R appuser:appuser /app /home/appuser /tmp/mcp-cache

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl --fail http://localhost:8000/health || exit 1

ENTRYPOINT ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]

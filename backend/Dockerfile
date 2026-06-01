# ==============================================================================
# RAG System — Dockerfile
# ==============================================================================
# Multi-stage build:
#   Stage 1 (builder): install Python deps into a venv
#   Stage 2 (runtime): copy venv + app code, run as non-root user
#
# This keeps the final image lean and the build layer cacheable.
# ==============================================================================

# ---- Stage 1: Builder -------------------------------------------------------
FROM python:3.11-slim AS builder

WORKDIR /build

# System deps needed to compile llama-cpp-python and pdfplumber
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    libgomp1 \
    ghostscript \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Stage 2: Runtime -------------------------------------------------------
FROM python:3.11-slim AS runtime

WORKDIR /app

# Runtime system libraries
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    ghostscript \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy venv from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application source
COPY app/ ./app/
COPY tests/ ./tests/

# Create data directories (mounted as volumes in production)
RUN mkdir -p data/pdfs data/index models

# Non-root user for security
RUN useradd -m -u 1001 raguser && chown -R raguser:raguser /app
USER raguser

# Expose FastAPI port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

# Startup command
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]

FROM python:3.12-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir fastapi uvicorn

# Copy application
COPY . /app/arn_v9/

# Pre-download the embedding model at build time
# so the container starts fast with no internet needed
RUN python3 -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Data volume
VOLUME /data
ENV ARN_DATA_ROOT=/data

# Expose API port
EXPOSE 8742

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8742/v1/health')" || exit 1

# Run with single worker (embedding model is ~90MB per process)
CMD ["python3", "-m", "uvicorn", "arn_v9.api.server:app", \
     "--host", "0.0.0.0", "--port", "8742", "--workers", "1"]

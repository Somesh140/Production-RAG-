# Enterprise Knowledge System - PROVIDED, COMPLETE (Activity 3.3).
# Participant work for 3.3 is `docker build`, `docker compose up`, and
# verifying the portal loads inside the container - not authoring this file.
FROM python:3.12-slim

WORKDIR /app

# Build deps for faiss-cpu / numpy / chromadb wheels on slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector index at image build time would need an API key, so it is
# built on first run instead (KnowledgeSystem does it lazily). The corpus and
# manifest are baked in.

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

CMD ["streamlit", "run", "app/knowledge_portal.py", "--server.port=8501", "--server.address=0.0.0.0"]

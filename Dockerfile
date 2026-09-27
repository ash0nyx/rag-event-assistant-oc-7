# Image for the Puls-Events RAG API.
#
# What goes in: Python, the locked dependencies, the code.
# What stays out: secrets (.env, passed at run time) and data/ (index and CSV,
# mounted as a volume so a rebuild does not need a new image).
#
# Build:  docker build -t puls-events-rag .
# Run:    docker run -p 8000:8000 --env-file .env -v ./data:/app/data puls-events-rag
# Or:     docker compose up --build

FROM python:3.12-slim

# uv, copied from its official image: installs the exact versions in uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /bin/uv

# curl is only needed by the health check.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first, code second: a code change reuses the cached install.
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project

COPY api/ api/
COPY rag/ rag/
COPY scripts/ scripts/
COPY README.md ./
RUN uv sync --frozen --no-dev

# Run as a non-root user. uid 1000 is the usual first user on Linux, so files
# written to the mounted data/ folder (by /rebuild) belong to the host user,
# not to root. Also limits the damage if the API is ever compromised.
RUN useradd --uid 1000 --create-home app && chown -R app:app /app
USER app

# Where the index lives inside the container (mount ./data here).
VOLUME ["/app/data"]

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 0.0.0.0: listen on every interface so the published port reaches the app.
CMD ["uv", "run", "--no-sync", "uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]

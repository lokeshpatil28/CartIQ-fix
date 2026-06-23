# Backend (FastAPI inference service).
# Build context is the REPO ROOT, not backend/, because the service imports
# the shared feature module from data_generation/ and the pipeline needs the
# whole tree:  docker build -f backend/Dockerfile -t cartiq-backend .
FROM python:3.12-slim

# libgomp1 is the OpenMP runtime LightGBM links against; the pip wheel does
# not bundle it, so the import fails at runtime without this.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first so this layer is cached unless requirements change.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Train the model from source AT BUILD TIME. This bakes a model.joblib into
# the image that is guaranteed to match the committed code — no stale-binary
# risk, no committed model artifact to drift from the source that produced
# it. The pipeline is fully seeded, so this is reproducible. Heavy work
# happens here (build), keeping the runtime instance light (it only loads the
# small model + items.csv and serves).
RUN python run_pipeline.py

# Render/most PaaS inject $PORT; default to 8000 for local/compose.
ENV PORT=8000
EXPOSE 8000

# Container-native health probe hits the existing /health endpoint.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -fsS "http://localhost:${PORT:-8000}/health" || exit 1

CMD ["sh", "-c", "uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8000}"]

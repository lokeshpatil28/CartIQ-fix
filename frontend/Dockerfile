# Frontend (Streamlit demo client).
# Build context is the REPO ROOT so this and the backend share one context:
#   docker build -f frontend/Dockerfile -t cartiq-frontend .
FROM python:3.12-slim

WORKDIR /app

# Lean deps (streamlit + requests only) — see frontend/requirements.txt.
COPY frontend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY frontend/ ./frontend/

# API_URL is read by frontend_app.py at runtime (os.environ). Default points
# at the compose service name; override it with the live backend URL on
# Streamlit Cloud / any deploy. No code change needed to repoint the backend.
ENV API_URL=http://backend:8000
EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8501/_stcore/health').read()==b'ok' else 1)"

CMD ["sh", "-c", "streamlit run frontend/frontend_app.py --server.port=8501 --server.address=0.0.0.0 --server.headless=true"]

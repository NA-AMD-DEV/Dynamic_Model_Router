# Judging VM is amd64. Build with --platform linux/amd64 (buildx on Apple Silicon).
FROM python:3.11-slim

# Unbuffered so stderr diagnostics survive a hard kill at the 10-minute wall.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Dependencies before source, so editing a prompt doesn't rebuild the wheel layer.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# No .env, no API key. FIREWORKS_BASE_URL, FIREWORKS_API_KEY, and ALLOWED_MODELS
# are injected by the harness at run time and read via os.environ.
ENTRYPOINT ["python", "-m", "agent.main"]

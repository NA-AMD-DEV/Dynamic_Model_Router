# Judging VM is amd64. Build with --platform linux/amd64 (buildx on Apple Silicon).
FROM python:3.11-slim

# PYTHONUNBUFFERED: stderr diagnostics survive a hard kill at the 10-minute wall.
# PIP_NO_CACHE_DIR / PYTHONDONTWRITEBYTECODE: no wheel cache, no .pyc in the layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Dependencies before source, so editing a prompt reuses the wheel layer instead
# of reinstalling. R2 changes prompts far more often than dependencies.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY agent/ ./agent/

# Runs as root deliberately: the harness owns the /output mount, and a non-root
# user risks EACCES on results.json — a missing output file scores zero, which
# is a far worse outcome than running privileged in a throwaway judging VM.

# No .env, no API key. FIREWORKS_BASE_URL, FIREWORKS_API_KEY, and ALLOWED_MODELS
# are injected by the harness at run time and read via os.environ.
ENTRYPOINT ["python", "-m", "agent.main"]

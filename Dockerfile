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

# Sequential by default for v3: eliminates the concurrency class of bugs that
# caused the v2 0% judging failure (thread-unsafe _UNAVAILABLE set).  The
# measured wall time at concurrency=1 is ~48s for 64 tasks — far under the
# 10-minute limit.  Thread-safety locks are in place (config.py, fireworks_client.py),
# so raising to 3 is safe once verified, but not worth the risk for v3.
ENV ROUTER_CONCURRENCY=1

# No .env, no API key. FIREWORKS_BASE_URL, FIREWORKS_API_KEY, and ALLOWED_MODELS
# are injected by the harness at run time and read via os.environ.
ENTRYPOINT ["python", "-m", "agent.main"]

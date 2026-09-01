# MedXplain API image.
#
# Model weights are NOT baked in: checkpoints are hundreds of megabytes and are
# produced by the training scripts, so results/checkpoints is mounted at run
# time. That keeps the image small and means a retrained model does not require
# an image rebuild.

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/app/.cache/huggingface \
    MEDXPLAIN_DEVICE=cpu

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential curl \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# CPU-only torch keeps the image at a fraction of the CUDA build's size.
COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r backend/requirements.txt

COPY medxplain ./medxplain
COPY backend ./backend
COPY configs ./configs

RUN useradd --create-home --uid 1000 app \
 && mkdir -p /app/results /app/data /home/app/.cache/huggingface \
 && chown -R app:app /app /home/app
USER app

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
  CMD curl -fsS http://localhost:8000/api/health || exit 1

CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000"]

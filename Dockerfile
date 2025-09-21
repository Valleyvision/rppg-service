# File: Dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN useradd -m -u 1001 appuser

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY app ./app
RUN mkdir -p /app/recordings && chown -R appuser:appuser /app

USER appuser
EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

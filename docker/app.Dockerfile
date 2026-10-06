# One image for the scoring API and the dashboard (different commands in compose).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOME=/tmp

# libgomp1: OpenMP runtime that XGBoost needs on slim images
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Code only. Data, models and reports are mounted at run time (see docker-compose.yml).
COPY src/ src/
COPY api/ api/
COPY dashboard/ dashboard/

EXPOSE 8000 8501

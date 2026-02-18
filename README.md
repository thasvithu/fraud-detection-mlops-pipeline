---
title: Fraud Detection MLOps API
emoji: 🚨
colorFrom: blue
colorTo: green
sdk: docker
app_port: 8000
---

# Fraud Detection MLOps Pipeline

End-to-end fraud detection project with reproducible training, model tracking, API serving, containerization, and CI.

## Project Status
Phases 0, 1, 2, 3, 4, 5, 6, 7, 8, and 9 are complete.

## Repository Structure
```
api/
src/
configs/
data/
models/
artifacts/
logs/
tests/
.github/workflows/
```

## Prerequisites
- Python 3.11+
- `uv`

## Setup
```bash
# You will create/activate the virtual environment.
uv pip install -r requirements.txt
```

## Run Commands
- Train pipeline:
```bash
uv run python -m src.train
```
- Run tests:
```bash
uv run pytest
```
- Run API:
```bash
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Data
Place the Kaggle credit card dataset at:
- `data/raw/creditcard.csv`

## Implementation Status
All planned phases (0-9) are complete.

## Quality Gates
- Test runner: `pytest` with coverage gates
- Coverage threshold: `>= 80%` across `src` and `api`
- CI enforces passing tests and coverage on every PR/push to `main`

## CI/CD Pipeline
- Workflow: `.github/workflows/ci.yml`
- Jobs:
  - `test`: installs dependencies and runs `pytest` with coverage gate
  - `build-image`: builds Docker image after tests pass
  - `deploy`: optional webhook trigger on pushes to `main`
- Optional secret for deploy job:
  - `DEPLOY_WEBHOOK_URL`

## Hugging Face Space Auto-Deploy
- Workflow: `.github/workflows/deploy-hf-space.yml`
- Trigger: push to `main` (and manual `workflow_dispatch`)
- Required GitHub Secrets:
  - `HF_TOKEN`: Hugging Face User Access Token (write permission)
  - `HF_SPACE_REPO`: `<username>/<space-name>` (example: `vithu/fraud-detection-mlops-api`)

## Hugging Face Keepalive
- Workflow: `.github/workflows/keepalive-hf-space.yml`
- Trigger: scheduled (Mon/Wed/Fri at 09:00 UTC) + manual run
- Required GitHub Secret:
  - `HF_SPACE_URL`: full Space URL (example: `https://thasvithu-fraud-detection-mlops-api.hf.space`)

## Monitoring and Operations
- Endpoint: `GET /metrics`
- Metrics exposed:
  - `total_requests`
  - `error_count`
  - `error_rate`
  - `total_predictions`
  - `fraud_predictions`
  - `fraud_prediction_rate`
  - `avg_latency_ms`
- Request observability:
  - `X-Request-ID` response header
  - `X-Process-Time-Ms` response header
  - structured JSON logs for request completion and prediction summaries

## Operational Playbook
- High error rate:
  - Trigger: `error_rate > 0.05` over recent traffic
  - Checks: model/preprocessor artifacts present, recent deploys, endpoint exceptions
- Latency regression:
  - Trigger: `avg_latency_ms` increasing abnormally
  - Checks: container CPU/memory pressure, batch size spikes, model loading issues
- Prediction behavior drift:
  - Trigger: sudden change in `fraud_prediction_rate`
  - Checks: input schema/data drift, threshold configuration drift, retrain recency

## Containerization
- Build image:
```bash
docker build -t fraud-detection-api:latest .
```
- Run container:
```bash
docker run --rm -p 8000:8000 fraud-detection-api:latest
```
- Run with compose:
```bash
docker compose up --build
```
- Verify health:
```bash
curl http://localhost:8000/health
```

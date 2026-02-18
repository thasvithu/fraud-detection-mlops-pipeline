# Fraud Detection MLOps Pipeline

End-to-end fraud detection project with reproducible training, model tracking, API serving, containerization, and CI.

## Project Status
Phases 0, 1, 2, 3, 4, 5, 6, 7, and 8 are complete.

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

## Next Implementation Phases
- Phase 9: Monitoring and operations

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

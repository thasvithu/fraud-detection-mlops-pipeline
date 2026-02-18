# Fraud Detection MLOps Pipeline

End-to-end fraud detection project with reproducible training, model tracking, API serving, containerization, and CI.

## Project Status
Phase 0 scaffold is complete.

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
- Python 3.10+
- `pip`

## Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run Commands
- Train pipeline:
```bash
python -m src.train
```
- Run tests:
```bash
pytest -q
```
- Run API:
```bash
uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Data
Place the Kaggle credit card dataset at:
- `data/raw/creditcard.csv`

## Next Implementation Phases
- Phase 1: Data ingestion + validation
- Phase 2: Preprocessing pipeline
- Phase 3: Training + MLflow tracking
- Phase 4: Evaluation + thresholding
- Phase 5+: API hardening, CI/CD, monitoring

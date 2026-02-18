# Fraud Detection MLOps Pipeline

End-to-end fraud detection project with reproducible training, model tracking, API serving, containerization, and CI.

## Project Status
Phases 0 and 1 are complete.

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
python -m src.train
```
- Run tests:
```bash
uv run pytest -q
```
- Run API:
```bash
uv run uvicorn api.app:app --reload --host 0.0.0.0 --port 8000
```

## Data
Place the Kaggle credit card dataset at:
- `data/raw/creditcard.csv`

## Next Implementation Phases
- Phase 2: Preprocessing pipeline
- Phase 3: Training + MLflow tracking
- Phase 4: Evaluation + thresholding
- Phase 5+: API hardening, CI/CD, monitoring

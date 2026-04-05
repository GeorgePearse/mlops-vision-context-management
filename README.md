# mlops-vision-context-management

Standalone Python package for agentic vision context management, active-learning simulation, and instance-segmentation orchestration.

This repo currently contains the `agentic_vision` package extracted from the larger Visia monorepo. The main surfaces are:

- `agentic_vision.gemini_agentic_vision`: Gemini code-execution vision client
- `agentic_vision.instance_segmentation`: DSPy ReAct segmentation program with Qwen, Gemini, SAM3, mask refinement, zoom, and human-input hooks
- `agentic_vision.object_memory`: object/background memory retrieval and persistence
- `agentic_vision.viewer_runtime`: file-backed run/event/artifact recorder for live or replay visualization
- `scripts/run_active_learning_curve.py`: offline Critic + QueryPolicy budget simulation
- `scripts/build_active_learning_jsonl.py`: DB-to-JSONL builder for the simulation harness

## Install

```bash
uv sync
```

## Basic usage

```bash
uv run python scripts/run_gemini_agentic_vision.py \
  --dataset-name <dataset> \
  --limit 5
```

Common env vars:

- `GEMINI_API_KEY`
- `PG_DATABASE_URL`
- GCS credentials such as `GOOGLE_APPLICATION_CREDENTIALS`

## Active-learning simulation

```bash
uv run python scripts/run_active_learning_curve.py \
  --input-jsonl /path/to/frames.jsonl \
  --output-csv /tmp/active_learning_curve.csv \
  --output-plot /tmp/active_learning_curve.png \
  --max-queries-per-frame 2 \
  --max-queries-total 200 \
  --budget-step 10 \
  --optimize-prompts
```

Build the JSONL input directly from Postgres:

```bash
uv run python scripts/build_active_learning_jsonl.py \
  --dataset-name <dataset_name> \
  --training-split val \
  --limit-frames 1000 \
  --output-jsonl /tmp/active_learning_frames.jsonl
```

## Viewer runtime

`agentic_vision.viewer_runtime` records:

- run metadata
- ordered JSONL events
- rendered image artifacts

It is designed to sit underneath an API/UI layer, but the runtime itself lives in this package.

## Tests

```bash
PYTHONPATH=. python -m pytest tests/test_instance_segmentation_refinement.py tests/test_viewer_runtime.py -q
```

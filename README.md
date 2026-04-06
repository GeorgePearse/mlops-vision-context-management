# mlops-vision-context-management

Standalone Python package for agentic vision context management, active-learning simulation, and instance-segmentation orchestration.

This repo currently contains the `agentic_vision` package extracted from the larger Visia monorepo. The main surfaces are:

- `agentic_vision.gemini_agentic_vision`: Gemini code-execution vision client
- `agentic_vision.instance_segmentation`: DSPy ReAct segmentation program with Qwen, Gemini, SAM3, mask refinement, zoom, and human-input hooks
- `agentic_vision.object_memory`: object/background memory retrieval and persistence
- `agentic_vision.viewer_runtime`: file-backed run/event/artifact recorder for live or replay visualization
- `viewer_api`: minimal FastAPI layer for live/replay viewer access
- `frontend/agentic-vision-viewer`: standalone React/Next.js viewer app
- `scripts/run_active_learning_curve.py`: offline Critic + QueryPolicy budget simulation
- `scripts/build_active_learning_jsonl.py`: DB-to-JSONL builder for the simulation harness

## Install

```bash
uv sync
```

## Tooling

Python type checks are handled with `ty`, and the viewer app uses `oxlint` in front of ESLint.

```bash
uv run ty check
cd frontend/agentic-vision-viewer && npm run lint
uv run prek run --all-files
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

For live viewer runs:

- `DASHSCOPE_API_KEY`
- optional `AGENTIC_VISION_VIEWER_RUNS_DIR`
- optional `QWEN_MODEL`

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

## Viewer API

Run the standalone FastAPI server:

```bash
uv run agentic-vision-viewer-api
```

Endpoints:

- `POST /agentic-vision-viewer/runs`
- `GET /agentic-vision-viewer/runs`
- `GET /agentic-vision-viewer/runs/{run_id}`
- `GET /agentic-vision-viewer/runs/{run_id}/events`
- `GET /agentic-vision-viewer/runs/{run_id}/stream`
- `GET /agentic-vision-viewer/runs/{run_id}/artifacts/{artifact_id}`

`frame_uri` accepts:

- local file paths
- `http://` / `https://` image URLs
- `gs://` URIs

Notes:

- the standalone repo does not bundle the SAM3 backend module
- camera-mask sqlc bindings are not bundled either
- those tool calls now fail with explicit messages instead of import crashes

## React viewer app

```bash
cd frontend/agentic-vision-viewer
npm install
NEXT_PUBLIC_AGENTIC_VISION_VIEWER_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## TODO

- [ ] Add an image of an x-ray we install

## Tests

```bash
PYTHONPATH=. python -m pytest \
  tests/test_instance_segmentation_refinement.py \
  tests/test_viewer_runtime.py \
  tests/test_viewer_api.py \
  tests/test_standalone_instance_segmentation_tools.py -q
```

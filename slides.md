---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #eee
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
  }
  h1, h2, h3 {
    color: #e94560;
  }
  a {
    color: #0f3460;
  }
  code {
    background: #16213e;
    color: #e94560;
  }
  table {
    font-size: 0.8em;
  }
  th {
    background: #16213e;
    color: #e94560;
  }
  td {
    background: #0f3460;
  }
  pre {
    background: #16213e !important;
  }
  blockquote {
    border-left: 4px solid #e94560;
    color: #ccc;
  }
---

# Dataset Compilation

### Treating annotation as a build system problem

---

## The Problem

We have mature tooling for **everything except the dataset**

- Experiment tracking — MLflow, W&B, etc.
- Model serving — TorchServe, Triton, etc.
- Training orchestration — Kubeflow, Ray, etc.

But how do you **assemble, validate, and version** the dataset itself?

> "How do you know your dataset is correct before you start training?"

---

## The Gap

Most teams build datasets by:

1. Manually exporting from an annotation tool
2. Running ad-hoc scripts to convert formats
3. Hoping nothing changed since last time
4. Debugging model failures that turn out to be data bugs

There's no **build system** for datasets.

---

## The Compile Analogy

Building a dataset is like compiling a large codebase:

**A DAG of specialized steps that can run in parallel**

Not a sequential pipeline.

---

## The Model Cascade

Different models = different cost/speed/quality tradeoffs

```
Raw Images
    │
    ├──▶ Fast detector (YOLO)        ──▶ bounding boxes
    │         cheap, runs on everything
    │
    ├──▶ SAM (prompted by detections) ──▶ segmentation masks
    │         moderate cost
    │
    └──▶ VLM / slow model            ──▶ verify high-value cases
              expensive, targeted
```

---

## Why This Works

Each stage is **independent per-image**

→ Massively parallelizable

Expensive models only touch a **fraction** of the data

→ Cost efficient

Like `make -j16` for your dataset.

---

## The Analogy

| Concept | Code Compilation | Dataset Compilation |
|---|---|---|
| Source files | `.c`, `.rs` files | Raw images |
| Compiler stages | Preprocessor → compiler → linker | Detector → segmenter → verifier |
| Parallelism | `make -j16` | Model stages run concurrently per-image |
| Incremental builds | Only recompile changed files | Only re-annotate new/changed images |
| Cost optimization | Optimize hot paths | Expensive models only where needed |

---

## The Annotation CLI

```bash
# Ingest from multiple sources
dataset ingest --source s3://raw-images --annotations coco.json

# Validate before you compile
dataset validate --check balance --check coverage

# Compile: orchestrate the model cascade
dataset compile --config pipeline.yaml --parallel 64

# What changed?
dataset diff v1.2 v1.3
```

CLI-first → fits into CI/CD and automation

---

## Visual Validation

### Segmentation vs Outline

Crop a subset of an object. Compare the **mask** against the **outline**.

This surfaces:
- Masks bleeding outside the object boundary
- Under-segmented regions
- Annotation drift across annotators

Hard to catch in aggregate stats.
**Obvious** when rendered side-by-side.

---

## The Pipeline Config

```yaml
# pipeline.yaml
stages:
  detect:
    model: yolov8-x
    run_on: all
    output: bounding_boxes

  segment:
    model: sam-2
    depends_on: detect
    run_on: all
    output: masks

  verify:
    model: gpt-4o
    depends_on: segment
    run_on:
      filter: confidence < 0.85 OR area > 50000
    output: verified_masks
```

Declarative. Version-controlled. Reproducible.

---

## Key Takeaways

1. **Dataset assembly is the missing build system** in MLOps
2. **Compile, don't script** — orchestrate a DAG of models
3. **Parallelize** — most work is embarrassingly parallel
4. **Optimize cost** — expensive models only where they matter
5. **Validate visually** — aggregate stats hide annotation bugs
6. **CLI-first** — automation and CI/CD from day one

---

## Thank You

**github.com/GeorgePearse/mlops-vision-context-management**

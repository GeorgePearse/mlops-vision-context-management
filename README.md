# mlops-vision-context-management

## Core Idea

Treat dataset creation as **compilation** — a repeatable, versioned build step rather than a one-off manual process.

## Focus: Annotation CLI

A command-line tool for managing the full lifecycle of vision annotation datasets:

- **Define** annotation schemas and label taxonomies from config
- **Ingest** raw images and existing annotations from multiple sources/formats (COCO, YOLO, Pascal VOC, etc.)
- **Transform** — filter, split, merge, remap labels, augment
- **Validate** — check annotation consistency, coverage, class balance
- **Compile** — produce a single, versioned, reproducible dataset artifact ready for training
- **Diff** — compare dataset versions, track what changed and why

## Why "Compile"?

Building a dataset is like compiling a large codebase — it's a **DAG of specialized steps that can run in parallel**, not a sequential pipeline.

### The Model Cascade

Different models have different cost/speed/quality tradeoffs. A compiled dataset orchestrates them:

```
Raw Images
    │
    ├──▶ Fast detector (e.g. YOLO) ──▶ bounding boxes     ← cheap, runs on everything
    │
    ├──▶ SAM (prompted by detections) ──▶ segmentation masks  ← moderate cost
    │
    └──▶ VLM / slow model ──▶ verify masks on high-value cases  ← expensive, targeted
```

Each stage is **independent per-image** — massively parallelizable. The "compile" step orchestrates which models run on what, routes outputs between stages, and assembles the final dataset artifact.

### The Analogy

| Concept | Code compilation | Dataset compilation |
|---|---|---|
| Source files | `.c`, `.rs` files | Raw images |
| Compiler stages | Preprocessor → compiler → linker | Detector → segmenter → verifier |
| Parallelism | `make -j16` | Run model stages across images concurrently |
| Incremental builds | Only recompile changed files | Only re-annotate new/changed images |
| Cost optimization | Optimize hot paths | Run expensive models only where needed |

The key insight: **most of the work is embarrassingly parallel**, and the expensive models only need to touch a fraction of the data. A good compilation strategy minimizes cost while maximizing annotation quality.

## Visual Validation: Segmentation vs Outline

A key validation step — crop a subset of an object and compare the **segmentation mask** against the **object outline**. This surfaces:

- Masks that bleed outside the actual object boundary
- Under-segmented regions where the mask misses parts of the object
- Annotation drift across frames or annotators

This kind of visual diff is hard to catch in aggregate stats but obvious when rendered side-by-side on a crop.

## Presentation Notes

- The gap in current MLOps tooling is at the **annotation management** layer — plenty of tools for experiment tracking and model serving, but dataset assembly is still ad-hoc
- Key audience question: "How do you know your dataset is correct before you start training?"
- Demo flow: show `ingest -> validate -> compile -> diff` as a tight CLI loop
- Contrast with GUI-only annotation platforms — CLI-first means it fits into CI/CD and automation

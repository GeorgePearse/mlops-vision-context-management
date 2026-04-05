---
marp: true
theme: default
paginate: true
backgroundColor: #1a1a2e
color: #eee
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    padding-left: 220px;
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
    font-size: 0.75em;
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
  .nav {
    position: absolute;
    top: 0;
    left: 0;
    width: 190px;
    height: 100%;
    background: #0f3460;
    padding: 30px 0 30px 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0;
    z-index: 10;
    border-right: 2px solid #e94560;
  }
  .nav span {
    display: block;
    padding: 6px 14px;
    font-size: 0.48em;
    color: #7a8ba6;
    line-height: 1.3;
  }
  .nav span.active {
    color: #fff;
    background: rgba(233, 69, 96, 0.25);
    border-left: 3px solid #e94560;
    font-weight: bold;
  }
---

<div class="nav">
<span class="active">About Me</span>
<span>Visia</span>
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

# George Pearse

### Machine vision, data quality, and annotation systems

- Working on practical vision workflows where dataset quality is often the bottleneck
- Interested in reproducible annotation, validation, and dataset compilation
- This talk is about making data preparation feel more like engineering

---

<div class="nav">
<span>About Me</span>
<span class="active">Visia</span>
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## Visia

- Computer vision work makes label quality, review loops, and data iteration core product concerns
- The hard part is often not training one more model but building confidence in the dataset
- The rest of this deck frames that problem as infrastructure: compile, validate, version, and diff the data

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span class="active">Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

# Dataset Compilation

### Treating annotation as a build system problem

---

<div class="nav">
<span>Dataset Compilation</span>
<span class="active">The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## The Problem

We have mature tooling for **everything except the dataset**

- Experiment tracking — MLflow, W&B, etc.
- Model serving — TorchServe, Triton, etc.
- Training orchestration — Kubeflow, Ray, etc.

But how do you **assemble, validate, and version** the dataset itself?

> "How do you know your dataset is correct before you start training?"

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span class="active">The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## The Gap

Most teams build datasets by:

1. Manually exporting from an annotation tool
2. Running ad-hoc scripts to convert formats
3. Hoping nothing changed since last time
4. Debugging model failures that turn out to be data bugs

There's no **build system** for datasets.

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span class="active">The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## The Compile Analogy

Building a dataset is like compiling a large codebase:

**A DAG of specialized steps that can run in parallel**

Not a sequential pipeline.

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span class="active">The Model Cascade</span>
<span>Why This Works</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

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

<!-- Note: optimize for bandwidth and communication between stages, especially across the Segment Anything family. Ask what the downstream model can be prompted with cheaply and reliably: boxes, inside/outside points, sparse clicks, prior masks. Prefer upstream steps that emit the most useful prompt signal with the least communication cost. -->

<!-- Note: compress context into dynamic retrieval between stages. Instead of passing full state forward, distill it into retrievable signals and only fetch richer context when confidence is too low or when a human confirmation step is required. -->

<!-- Note: there is a large test-time inference / chain-of-thought tuning space here. Two main axes are how specific each question is and how small a crop you inspect. Narrower questions over tighter crops can often improve results. -->

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span class="active">Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## Why This Works

Each stage is **independent per-image**

→ Massively parallelizable

Expensive models only touch a **fraction** of the data

→ Cost efficient

Like `make -j16` for your dataset.

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span class="active">Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## What Are Your Degrees of Freedom?

- Where is the capacity in the agent to actually adapt at inference time?
- The main knobs are prompt structure, retrieved context, question specificity, and crop size
- Prefer **knowledge consolidation**: pull things out of the active prompt, but keep them accessible via retrieval or search
- Contrast that with **compaction**: just shrinking the prompt and losing access to the underlying information
- For segmentation, learning through prompts and context is only just about possible right now
- So be explicit about what must come from the base model versus what you can steer at test time

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span class="active">The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## The Analogy

| Concept | Code Compilation | Dataset Compilation |
|---|---|---|
| Source files | `.c`, `.rs` files | Raw images |
| Compiler stages | Preprocessor → compiler → linker | Detector → segmenter → verifier |
| Parallelism | `make -j16` | Model stages run concurrently per-image |
| Incremental builds | Only recompile changed files | Only re-annotate new/changed images |
| Cost optimization | Optimize hot paths | Expensive models only where needed |

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span class="active">The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

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

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span class="active">Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

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

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span class="active">The Pipeline Config</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

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

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>Degrees of Freedom</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span class="active">Key Takeaways</span>
<span>Thank You</span>
</div>

## Key Takeaways

1. **Dataset assembly is the missing build system** in MLOps
2. **Compile, don't script** — orchestrate a DAG of models
3. **Parallelize** — most work is embarrassingly parallel
4. **Optimize cost** — expensive models only where they matter
5. **Validate visually** — aggregate stats hide annotation bugs
6. **CLI-first** — automation and CI/CD from day one

---

<div class="nav">
<span>Dataset Compilation</span>
<span>The Problem</span>
<span>The Gap</span>
<span>The Compile Analogy</span>
<span>The Model Cascade</span>
<span>Why This Works</span>
<span>The Analogy</span>
<span>The Annotation CLI</span>
<span>Visual Validation</span>
<span>The Pipeline Config</span>
<span>Key Takeaways</span>
<span class="active">Thank You</span>
</div>

## Thank You

**github.com/GeorgePearse/mlops-vision-context-management**

<!-- Add a link at the end to the DSPy article: if you don't use DSPy, you build DSPy, and you should only build it if you first know and understand DSPy. -->

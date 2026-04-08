---
marp: true
theme: default
paginate: true
backgroundColor: #f5f3ef
color: #2d2d2d
style: |
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding-left: 220px;
  }
  h1, h2, h3 {
    color: #1a1a1a;
  }
  a {
    color: #3a7d7e;
  }
  code {
    background: #e8e5e0;
    color: #3a7d7e;
  }
  table {
    font-size: 0.75em;
  }
  th {
    background: #2d2d2d;
    color: #f5f3ef;
  }
  td {
    background: #e8e5e0;
    color: #2d2d2d;
  }
  pre {
    background: #2d2d2d !important;
    color: #f5f3ef !important;
  }
  blockquote {
    border-left: 4px solid #3a7d7e;
    color: #555;
  }
  strong {
    color: #3a7d7e;
  }
  .nav {
    position: absolute;
    top: 0;
    left: 0;
    width: 190px;
    height: 100%;
    background: #2d2d2d;
    padding: 30px 0 30px 0;
    display: flex;
    flex-direction: column;
    justify-content: center;
    gap: 0;
    z-index: 10;
    border-right: 2px solid #3a7d7e;
  }
  .nav span {
    display: block;
    padding: 6px 14px;
    font-size: 0.48em;
    color: #888;
    line-height: 1.3;
  }
  .nav span.active {
    color: #f5f3ef;
    background: rgba(58, 125, 126, 0.2);
    border-left: 3px solid #3a7d7e;
    font-weight: bold;
  }
---

<div class="nav">
<span class="active">About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

![bg right:35% contain](images/george_linked_in_pic.jpg)

## About Me

- Started as a Data Engineer
- A few years in a medical imaging start-up working on lung cancer detection in x-rays, and brain tumours in CT scans
- Approaching 4 years at Visia, building Computer Vision applications for recyling and heavy industry.

---

<div class="nav">
<span>About Me</span>
<span class="active">Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## Visia

- Started off selling cameras to recycling facilities
- Chucked x-rays in for good measure (and preventing fires from Lithium Ion Batteries)
- Added Lidar into the mix for some applications
- Now Multimsensor AI for Heavy Industry

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span class="active">What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## What we Do

![bg right:55% contain](images/instance-segmentation-demo.png)

Help with cost disagreements between buyers and sellers at metal yards. 

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span class="active">What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## What we Do

![bg right:55% contain](images/xray-detection-demo.png)

Find batteries in e-waste and municipals recycling with x-rays and lasers.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span class="active">What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## What we Do

![bg right:55% contain](images/aerial-detection-demo.png)

Detect and send notifications for 'bulkies' in waste to energy facilities.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span class="active">What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## What is an Agent?

A working definition:

- **Runs continuously** or for a non-predetermined amount of time
- **Takes human input** at some point in its lifecycle

A closed system with fixed inputs and deterministic termination is **not an agent** — it's a pipeline.

| System | Human input? | Open-ended runtime? | Agent? |
|--------|--------------|---------------------|--------|
| Batch inference script | No | No | No |
| GEPA optimization loop | Yes (seed, feedback) | Yes (until convergence) | Yes |
| Active learning annotator | Yes (corrections) | Yes (ongoing) | Yes |
| One-shot VLM call | No | No | No |

The distinction matters: agents need **context management** across time. Pipelines just need correct wiring.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span class="active">Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## Types of Context

- **Prompt context** — system instructions and framing baked into the initial call
- **Retrieved context (RAG)** — information pulled from external stores at query time
- **Conversational context** — the history of the current interaction
- **Tool/observation context** — results from tool calls and API responses the agent generates for itself
- **Visual context** — composite images encoding predictions, ground truth, or crops for a downstream model
- **Persistent context** — memories and session state carried across conversations (the cookies of agents, but more like a cheatsheet intentionally kept small)

<!-- The core tension: context management is compression and selection. Finite window, right information, right time. Too little → hallucination. Too much → noise. -->

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span class="active">GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## GEPA Algorithm

GEPA's core algorithm iterates through three stages — **Executor**, **Reflector**, **Curator** — each with distinct context demands.

| Stage | Role | Context |
|-------|------|---------|
| **Executor** | Runs candidate program on a small training minibatch (`reflection_minibatch_size`, default 3 examples) | Captures full execution traces: reasoning chains, intermediate outputs, tool calls, error messages |
| **Reflector** | Feeds traces + evaluator feedback into a strong LLM (`reflection_lm`) | Diagnoses failure modes and identifies causal patterns |
| **Curator** | Proposes concrete instruction mutation based on the diagnosis | Transforms reflection into actionable prompt edits |

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span class="active">GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## GEPA Context Management

GEPA (Generalized Evolutionary Prompt Adaptation) manages context through:

- **Evaluation cache** — stores `(candidate, example)` results to avoid redundant inference
- **Reflective dataset** — captures execution traces for reflection-based prompt mutation
- **Pareto frontiers** — tracks best programs per validation example or objective (compressed historical context)
- **Batch sampling** — strategic minibatch selection balances coverage vs. cost
- **State persistence** — `GEPAState` serializes candidate evolution, enabling resumable optimization

The key insight: **consolidate context into retrievable signals** rather than passing full state forward. Only fetch richer context when confidence is low.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span class="active">Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>Thank You</span>
</div>

## Handling Context Pressure

Practical mitigations for the double-pressure problem:

- **Reduce `reflection_minibatch_size`** from 3 to 1–2 for ReAct programs with long trajectories
- **Use a high-context reflection LM** — models with large context windows (10M tokens ideal)
- **Reduce ReAct's `max_iters`** to 3–5 instead of 20
- **Keep tool return values concise** — control retrieved passage counts and output verbosity
- **Override `truncate_trajectory()`** for domain-aware truncation that preserves the most informative steps, not just the most recent

**Proven architecture for GEPA+ReAct**: Individually optimized specialist ReAct subagents (each with domain-specific tools) combined under a lead routing agent, with GEPA applied to each component separately before optimizing the coordinator.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span class="active">Visual Validation</span>
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
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span class="active">Key Takeaways</span>
<span>Thank You</span>
</div>

## Key Takeaways

1. **Context management is the core challenge** for vision agents
2. **Consolidate, don't accumulate** — compress context into retrievable signals
3. **Visual context matters** — images preserve raw signal that text loses
4. **Validate visually** — aggregate stats hide annotation bugs
5. **Know your degrees of freedom** — be explicit about what the model can learn at test time

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span class="active">Thank You</span>
</div>

## Thank You

**github.com/GeorgePearse/vision-agents**

### We're Hiring

If this kind of work interests you, reach out.

<!-- Add a link at the end to the DSPy article: if you don't use DSPy, you build DSPy, and you should only build it if you first know and understand DSPy. -->

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
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span class="active">What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
<span>Thank You</span>
</div>

## What we Do

![bg right:55% contain](images/xray-detection-demo.png)

Finding batteries in e-waste and municipals recycling with x-rays and lasers.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span class="active">What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
<span>Thank You</span>
</div>

## What we Do

![bg right:55% contain](images/aerial-detection-demo.png)

Detecting and sending notifications for 'bulkies' in waste to energy facilities.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span class="active">What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span class="active">GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span>GEPA Context</span>
<span class="active">Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
<span>Thank You</span>
</div>

## Image vs Text: Choosing Your Canvas

When passing context between pipeline stages, you have a choice:

| Approach | Pros | Cons |
|----------|------|------|
| **Image** (render predictions onto pixels) | Downstream model can re-interpret; errors don't propagate as hard | Vision understanding is weaker; lossy |
| **Text/JSON** (structured output) | LLMs understand text far better; precise; composable | Early mistakes propagate; no second chance to see the raw signal |

**The trade-off**: Text lets you reason precisely, but if stage 1 misreads "7" as "1", stage 2 has no way to recover. An image overlay at least preserves the original pixels for reinterpretation.

**Hybrid approach**: Pass both — structured JSON for the happy path, but include a crop or thumbnail so the model can sanity-check ambiguous cases.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Context</span>
<span>Image vs Text</span>
<span class="active">Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span class="active">Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
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
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span class="active">Key Takeaways</span>
<span>References</span>
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
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span class="active">References</span>
<span>Thank You</span>
</div>

## References

- [Contextual Retrieval](https://www.anthropic.com/engineering/contextual-retrieval) — Anthropic's approach to improving RAG with contextual embeddings and contextual BM25
- [Late Interaction Overview](https://weaviate.io/blog/late-interaction-overview) — Weaviate's overview of late interaction models for retrieval

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>What is an Agent?</span>
<span>Types of Context</span>
<span>GEPA Context</span>
<span>Image vs Text</span>
<span>Degrees of Freedom</span>
<span>Visual Validation</span>
<span>Key Takeaways</span>
<span>References</span>
<span class="active">Thank You</span>
</div>

## Thank You

**github.com/GeorgePearse/vision-agents**

### We're Hiring

If this kind of work interests you, reach out.

<!-- Add a link at the end to the DSPy article: if you don't use DSPy, you build DSPy, and you should only build it if you first know and understand DSPy. -->

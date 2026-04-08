---
marp: true
theme: default
paginate: true
backgroundColor: #f5f3ef
color: #2d2d2d
style: |
  section {
    font-family: 'Inter', 'Segoe UI', sans-serif;
    padding-left: 250px;
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
    width: 220px;
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
  .nav span.sub {
    padding-left: 28px;
    font-size: 0.42em;
  }
  .profile-pic {
    float: right;
    width: 180px;
    height: 180px;
    border-radius: 50%;
    object-fit: cover;
    margin: 0 0 20px 30px;
    border: 3px solid #3a7d7e;
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
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## About Me

<img src="images/george_linked_in_pic.jpg" class="profile-pic" />

- Started as a Data Engineer
- A few years in a medical imaging start-up working on lung cancer detection in x-rays, and brain tumours in CT scans
- Approaching 4 years at Visia, building Computer Vision applications for recyling and heavy industry.

---

<div class="nav">
<span>About Me</span>
<span class="active">Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## Visia

- **Multisensor AI for Heavy Industry** — cameras, x-rays, and lidar
- Started selling cameras to recycling facilities, added x-rays for detecting Lithium Ion Batteries (fire prevention), then lidar
- We help recyclers, metal yards, and waste-to-energy plants make faster, more accurate decisions
- ~30 people, based in Dublin with customers across Europe

---

<div class="nav">
<span>About Me</span>
<span class="active">Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## Visia

![bg right:55% contain](images/office.jpeg)

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span class="active">What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
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
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
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
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
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
<span class="active">Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## Where do we use Agents

- **Billing disagreements in metal yards**
- **Active learning simulations**

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub active">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## Metal Yard Application

![bg right:55% contain](images/metal-yard-photo.png)

<!-- TODO: Add photos of metal yard dumps -->

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub active">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## Metal Yard Application — 2. Program Tools

- The agent can **zoom in** on an uncertain or dense region
- The agent has both an **image-level** and **object-level** store
- Some simple dumps only require image-level retrieval (all the same material), most are more complicated and have a huge mix of materials made of different materials
- It can use **SAM3** to get the area of an object
- All in service of finding the right **material grade** and the **cost deduction** — the accuracy of these 2 outputs are the core of the metric, with just a little bit of reward for accurate descriptions because we think it improves the product UX. The output can be fed back into the start of the program

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub active">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## Metal Yard Application — 3. Tech Stack

- **Turbopuffer** for the object memory and image memory
- **DINOv3** for the embeddings
- **Gemini** to coordinate the system
- **Qwen3.5** for the initial boxes and when to zoom
- **SAM3** for masks when needed

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub active">4. Context Management</span>
<span>Types of Context</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Thank You</span>
</div>

## Metal Yard Application — 4. Context Management

- When optimising this program with GEPA, we can't actually keep the image within the context of the reflection LM, because it almost instantly exceeds the token limit
- GEPA keeps multiple instances of your dataset within the context of the reflection LLM, in order to work out how to optimise across them

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span class="active">GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
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
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span class="active">GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## GEPA Context Management

GEPA (Generalized Evolutionary Prompt Adaptation) manages context through:

- **Evaluation cache** — stores `(candidate, example)` results to avoid redundant inference
- **Reflective dataset** — captures execution traces for reflection-based prompt mutation
- **Pareto frontiers** — tracks best programs per validation example or objective (compressed historical context)
- **Batch sampling** — strategic minibatch selection balances coverage vs. cost
- **State persistence** — `GEPAState` serializes candidate evolution.

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span class="active">Context Pressure</span>
<span>Types of Context</span>
<span>Thank You</span>
</div>

## Handling Context Pressure

Practical mitigations for the double-pressure problem:

- **Reduce `reflection_minibatch_size`** from 3 to 1–2 for ReAct programs with long trajectories
- **Use a high-context reflection LM** — models with large context windows (10M tokens ideal)
- **Reduce ReAct's `max_iters`** to 3–5 instead of 20
- **Keep tool return values concise** — control retrieved passage counts and output verbosity
- **Override `truncate_trajectory()`** for domain-aware truncation that preserves the most informative steps, not just the most recent


---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span class="active">Types of Context</span>
<span>Thank You</span>
</div>

## Types of Context

- **Prompt context** — system instructions and framing baked into the initial call
- **Retrieved context (RAG)** — information pulled from external stores at query time
- **Conversational context** — the history of the current interaction
- **Tool/observation context** — results from tool calls and API responses the agent generates for itself
- **Persistent context** — memories and session state carried across conversations (the cookies of agents, but more like a cheatsheet intentionally kept small)

<!-- The core tension: context management is compression and selection. Finite window, right information, right time. Too little → hallucination. Too much → noise. -->

---

<div class="nav">
<span>About Me</span>
<span>Visia</span>
<span>What we Do</span>
<span>Where do we use Agents</span>
<span>Metal Yard Application</span>
<span class="sub">1. Photos</span>
<span class="sub">2. Program Tools</span>
<span class="sub">3. Tech Stack</span>
<span class="sub">4. Context Management</span>
<span>GEPA Algorithm</span>
<span>GEPA Context</span>
<span>Context Pressure</span>
<span>Types of Context</span>
<span class="active">Thank You</span>
</div>

## Thank You

Be kind to each other, AI's getting wild.

### We're Hiring

If this kind of work interests you, reach out.

<!-- Add a link at the end to the DSPy article: if you don't use DSPy, you build DSPy, and you should only build it if you first know and understand DSPy. -->

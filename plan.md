# Jev-Lite: A Mini "System One" Structured Decision Model

## 1. Motivation

Most ML mini-projects (RNN generators, CNN classifiers, transformer fine-tunes) produce **text or a single label**. TypeSafe's Jev (launched Sep 15, 2026) introduced a different paradigm: a **System 1 model** that skips text generation entirely and returns **typed, calibrated, machine-readable decisions** over multiple questions in a single parallel pass — no prompting, no parsing, no hallucination risk.

This project builds a small, faithful version of that idea: **one shared encoder + multiple typed decision heads**, trained with a **calibration-aware loss**, evaluated not just on accuracy but on how trustworthy its confidence scores are.

**Deliverable:** a working "ticket triage" decision model that takes a support ticket's text + metadata and returns a structured JSON of typed answers with calibrated confidence — plus a benchmark comparing it to a naive "prompt an LLM and regex-parse the output" baseline.

---

## 2. Scope (what "typed decisions" means here)

Three typed question primitives, matching Jev's spec:

| Type     | Description                                  | Example                               |
|----------|-----------------------------------------------|----------------------------------------|
| `choice` | Pick one of N fixed options                  | `queue`: billing / technical / account / other |
| `score`  | A continuous 0–1 confidence/severity value    | `severity`: 0.0–1.0                    |
| `null`   | Is this question even answerable given input? | `has_refund_request`: yes/no/unclear   |

Each question gets its own small output head on top of a shared state encoder. All heads fire in one forward pass — no sequential decoding.

---

## 3. Dataset

**Task:** Support ticket triage (mirrors TypeSafe's own example use case).

- **Source:** Kaggle "Customer Support Ticket" datasets (e.g., search "support ticket classification dataset"), or a synthetic set generated with an LLM if a good public one isn't found.
- **Fields needed per ticket:**
  - `text` (subject + body)
  - `is_urgent` (yes/no) — label
  - `queue` (billing/technical/account/other) — label
  - `severity` (0–1, can be derived from priority tags: low=0.2, medium=0.5, high=0.8, critical=1.0)
- **Size target:** 2,000–5,000 tickets is plenty. Split 80/10/10 train/val/test.
- **Fallback if no good labeled dataset exists:** generate ~1,000 synthetic tickets by templating (ticket topic × urgency × sentiment) — fast and gives full label control, which also makes calibration easier to evaluate cleanly.

---

## 4. Architecture

```
Ticket text
     │
     ▼
[Sentence Encoder]   ← MiniLM (sentence-transformers) or TF-IDF+SVD for speed
     │
     ▼
[Shared MLP trunk]   (e.g. 384 → 256 → 128, ReLU, dropout)
     │
     ├──► [is_urgent head]  → 2-class softmax
     ├──► [queue head]      → 4-class softmax
     └──► [severity head]   → 1 sigmoid output (regression)
```

- Keep the shared trunk small (this is a mini-project, not a research paper).
- Each head is just a `Linear` layer (or Linear→ReLU→Linear) off the shared trunk — this is what gives the "parallel evaluation" property: one encoder pass, N heads fire simultaneously.

---

## 5. Training

- **Loss:** weighted sum of per-head losses
  - `is_urgent`, `queue`: CrossEntropyLoss
  - `severity`: MSELoss (or BCE if you bucket it)
  - Total = `loss_urgent + loss_queue + 0.5 * loss_severity` (tune weights if one head dominates)
- **Optimizer:** Adam, lr=1e-3, ~15–20 epochs (small model, will converge fast)
- **Calibration step (the important, distinctive part):**
  - After standard training, check if predicted confidences match actual accuracy (a model that says "90% confident" should be right ~90% of the time)
  - Apply **temperature scaling**: learn a single scalar `T` post-hoc on the validation set that divides the logits before softmax, to sharpen/soften confidence
  - Plot a **reliability diagram** (predicted confidence bucket vs. actual accuracy) before and after calibration

---

## 6. Inference / Output Format

Given a new ticket, one forward pass returns:

```json
{
  "is_urgent": {"answer": "yes", "confidence": 0.92},
  "queue": {"answer": "technical", "confidence": 0.81},
  "severity": {"answer": 0.74}
}
```

No text generation, no JSON parsing of an LLM's free-form output — the structure is baked into the model's output heads.

---

## 7. Benchmark (this is your report's centerpiece)

Compare Jev-lite against a **baseline: prompt a small LLM + regex/JSON-parse its response** for the same 3 questions, on the same test set.

Measure:
| Metric                     | Jev-lite | LLM-prompt baseline |
|-----------------------------|----------|----------------------|
| Latency per ticket (ms)     |          |                      |
| Accuracy (is_urgent, queue) |          |                      |
| MAE (severity)              |          |                      |
| Calibration error (ECE)     |          |                      |
| Parse failure rate          | 0% (n/a) | (LLMs occasionally return malformed JSON) |

This table is what makes the project interesting beyond "I trained a classifier" — it demonstrates *why* the System 1 paradigm exists: speed + reliability + zero parsing failures, at the cost of needing predefined question types instead of open-ended text.

---

## 8. Milestones & Time Budget (~4–5 hrs total)

| Step | Task | Time |
|------|------|------|
| 1 | Get/generate dataset, clean labels | 45 min |
| 2 | Build sentence encoder pipeline (MiniLM embeddings) | 20 min |
| 3 | Build multi-head model + training loop | 45 min |
| 4 | Train, tune loss weights | 45 min |
| 5 | Temperature scaling + reliability diagram | 30 min |
| 6 | Build LLM-prompt baseline for comparison | 40 min |
| 7 | Run benchmark, make comparison table/plots | 30 min |
| 8 | Write up report (architecture diagram, results, discussion) | 45 min |

---

## 9. Stretch Goals (if time allows)

- Add a 4th question type demo: a `null`-type head like `"is_refund_request_clear": yes/no/unclear`
- Swap MiniLM for a from-scratch small transformer encoder trained only on this dataset — shows you can build the whole stack, not just the heads
- Wrap it in a tiny FastAPI endpoint that takes ticket JSON in and returns typed decisions out, mimicking how Jev is actually served
- Add a "confidence-gated escalation": if any head's confidence < 0.5, flag ticket for human review instead of acting automatically — a realistic product pattern for these models

---

## 10. Report Structure

1. Introduction — what Jev/System 1 models are, why they're different from LLMs
2. Architecture — diagram of shared encoder + typed heads
3. Dataset & preprocessing
4. Training & calibration methodology
5. Results — accuracy, calibration (reliability diagrams), benchmark vs. LLM baseline
6. Discussion — tradeoffs (fixed question set vs. flexibility, speed vs. expressiveness)
7. Conclusion & stretch goals attempted

# Jev-Lite: A Mini System-One Structured Decision Model

Jev-Lite is a lightweight, calibrated, multi-task deep learning architecture inspired by TypeSafe's **Jev** (Sep 2026) System-1 decision paradigm.

Instead of generating free-form natural language with generative LLMs and parsing markdown/JSON via regex, Jev-Lite consumes text state directly (support tickets + metadata) and executes **parallel typed decision heads** in a single forward pass (~1.16 ms latency, 0% parse failure rate).

---

## Key Highlights

- **Single Forward Pass (<1.2ms)**: TF-IDF (5k n-grams) + TruncatedSVD (128-d) encoder with a shared multi-layer trunk driving 4 typed decision heads simultaneously.
- **Typed Judgment Heads**:
  - `is_urgent`: Boolean choice (`yes` / `no`).
  - `queue`: Multi-class routing (`billing` / `technical` / `account` / `other`).
  - `severity`: Regression continuous score (`0.0 - 1.0`).
  - `refund_clear`: Null-type tri-state classification (`yes` / `no` / `unclear`).
- **Post-Hoc Temperature Calibration**: Temperature scaling tuned via L-BFGS on validation NLL to provide reliable, well-calibrated confidence scores.
- **Escalation-Gated Inference**: Confidence-aware routing automatically flags low-certainty decisions (`confidence < threshold`) to human operators.
- **Production-Ready FastAPI Server**: Web UI demo and REST API endpoints (`/decide`, `/health`).

---

## Architecture

```
Ticket text
     |
[TF-IDF(5000, 1-2gram) + SVD-128]
     |
[Shared MLP 128 -> 256 -> 128, ReLU, dropout 0.2]
     |---> urgent head    Linear(128 -> 2)  (choice: yes/no)
     |---> queue head     Linear(128 -> 4)  (choice: billing/technical/account/other)
     |---> severity head  Linear(128 -> 1) + sigmoid (score: 0-1)
     +---> refund head    Linear(128 -> 3)  (null-type: yes/no/unclear)
```

**Total Parameters:** ~70k parameters (Runs blazingly fast on CPU).

---

## Benchmark Comparison (Test Set, N=300)

| Metric | Jev-Lite (Ours) | Classical LogReg | Rule Prompt-Proxy | LLM API (Lit. Est.) |
|---|---|---|---|---|
| **acc is_urgent** | **0.887** | 0.877 | 0.877 | ~0.89 |
| **acc queue** | **1.000** | 1.000 | 0.880 | ~0.94 |
| **MAE severity** | **0.178** | 0.180 | 0.201 | ~0.19 |
| **ECE urgent** | **0.033** | 0.080 | N/A | ~0.12 - 0.18 |
| **Latency (ms/ticket)** | **1.16 ms** | 2.71 ms | 0.03 ms | 600 - 1200 ms |
| **Parse Fail Rate** | **0.00%** | 0.00% | ~2.7% | 2.0% - 5.0% |

> *Throughput (batched): ~53,000 tickets/sec on CPU.*

---

## Project Structure

```
├── app.py              # FastAPI server with built-in interactive demo UI
├── infer.py            # Standalone inference demo with confidence-gated escalation
├── model.py            # PyTorch JevLite multi-head neural network architecture
├── train.py            # Multi-task loss training & temperature scaling calibration
├── dataset.py          # Synthetic dataset generator & feature extraction pipeline
├── benchmark.py        # Benchmark suite evaluating Jev-Lite vs baselines
├── report.md           # Detailed experiment & architectural report
├── data/               # Dataset artifacts (tickets.csv)
├── artifacts/          # Serialized models, vectorizers, SVD, and calibration plots
└── requirements.txt    # Python dependencies
```

---

## Quickstart

### 1. Installation

Clone the repository and install requirements:

```bash
git clone https://github.com/Adarsh041-arch/JEV-Decision-model-experimental.git
cd JEV-Decision-model-experimental
pip install -r requirements.txt
```

### 2. Run Inference

Test single-pass structured inference on sample customer tickets:

```bash
python infer.py
```

Sample output:
```json
{
  "is_urgent": {
    "answer": "yes",
    "confidence": 0.972
  },
  "queue": {
    "answer": "technical",
    "confidence": 1.0
  },
  "severity": {
    "answer": 0.824
  },
  "refund_clear": {
    "answer": "no",
    "confidence": 0.811
  },
  "escalate_to_human": false
}
```

### 3. Start Interactive Web Demo & API

Launch the FastAPI application:

```bash
uvicorn app:app --reload
```

- **Interactive UI**: Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
- **REST Endpoint**:
  ```bash
  curl -X POST "http://127.0.0.1:8000/decide" \
       -H "Content-Type: application/json" \
       -d '{"text": "Production server down! Urgent assistance required", "gate": 0.5}'
  ```

---

## Pipeline Replication

To regenerate data, retrain the model, and run full benchmarks from scratch:

```bash
# 1. Generate tickets dataset
python dataset.py

# 2. Train multi-task trunk + calibrate temperature parameters
python train.py

# 3. Run comparative benchmark
python benchmark.py
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.

# Differential Privacy Dp Sgd

> **Domain:** Clinical Decision Support & Biomedical Computing  
> **Reference Guidelines & Standards:** `Standard Clinical Formulations & ISO/IEC Quality Frameworks`

<div align="center">

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB.svg?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-009688.svg?logo=fastapi&logoColor=white)
![Audit Trail](https://img.shields.io/badge/Audit-HMAC--SHA256_Tamper--Evident-brightgreen.svg)
![Zero-PHI Guard](https://img.shields.io/badge/Guard-Zero--PHI_Outbound-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?logo=docker&logoColor=white)

</div>

---

## 📖 What It Does

**Differential Privacy Dp Sgd** is an advanced analytical and computational platform implementing Rényi Differential Privacy (RDP) accountant & Gaussian noise perturbation engine.

---

## ⚙️ Key Capabilities & Algorithmic Modules

- **Deterministic Calculation Engine**: Strict compliance with standard reference formulations and thresholds.
- **Risk & Urgency Classification**: Multi-tier categorization with automated clinical/operational action recommendations.
- **Validation & Guardrails**: Rigorous input bounds checking and anomaly detection.

---

## 💻 CLI Quickstart & Usage

### 1. Gaussian Mechanism
```bash
python cli.py gaussian --value 100.0 --sensitivity 1.0 --epsilon 1.0 --delta 1e-5
```

### 2. Laplace Mechanism
```bash
python cli.py laplace --value 50.0 --sensitivity 1.0 --epsilon 1.0
```

### 3. Noise Scale Calculation
```bash
python cli.py noise-scale --mechanism gaussian --sensitivity 1.0 --epsilon 1.0 --delta 1e-5
```

### 4. Privacy Budget Composition
```bash
python cli.py composition --type basic --epsilon 0.1 --k 10
python cli.py composition --type advanced --epsilon 0.1 --delta 1e-6 --k 10 --delta-prime 1e-5
python cli.py composition --type rdp --epsilon 1.0 --delta 1e-5 --k 100 --sigma 1.5
```

### 5. Sensitivity Analysis
```bash
python cli.py sensitivity --query count
python cli.py sensitivity --query sum --lower 0 --upper 100
python cli.py sensitivity --query mean --lower 0 --upper 100 --n 50
```

### 6. DP-SGD Step
```bash
python cli.py dp-sgd --clip-norm 1.0 --epsilon 1.0 --delta 1e-5 --batch-size 32 --dim 10
```

### 7. Moments Accountant
```bash
python cli.py accountant --sigma 1.5 --steps 1000 --delta 1e-5
```

### 8. Privacy Budget Tracking
```bash
python cli.py budget --total-epsilon 10.0 --total-delta 1e-5 --spend-epsilon 1.0 --description "Query 1"
```

---

## 🛡️ Security & Enterprise Architecture

* **Zero-PHI Outbound Interceptor:** Active AST and regex inspection blocking SSNs, MRNs, phone numbers, and patient identifiers.
* **Tamper-Evident HMAC-SHA256 Audit Trail:** Chained, cryptographically signed logs for every evaluation and state transition.
* **Air-Gapped LLM Reasoning Adapter:** Agnostic integration for local Ollama instances (`llama3`, `mistral`), Claude 3.5 Sonnet, GPT-4o, and deterministic test mocks.
* **Active Learning Bayesian Calibration:** Dynamic tracker updating worker reliability weights and monitoring Brier calibration drift.
* **FastAPI & Prometheus Telemetry:** Exposes OpenAPI 3.1 REST endpoints and operational Prometheus metrics (`/metrics`).

---

## 📦 Installation

```bash
# Core library (no external dependencies needed for engine)
pip install -e .

# With web API support
pip install -e ".[web]"

# With development dependencies
pip install -e ".[dev]"
```

Set the audit secret key for persistent HMAC integrity:

```bash
export AUDIT_SECRET_KEY="your-secure-random-key-here"
```

---

## 🧪 Testing & Verification

Run the automated test suite:

```bash
pytest -v
```

Run with coverage:

```bash
pytest -v --tb=short
```

Execute the simulation benchmark:

```bash
python simulator.py 1000
```

---

## 🐳 Container Deployment

```bash
docker build -t differential-privacy-dp-sgd .
docker run -p 8000:8000 -e AUDIT_SECRET_KEY=your-secret differential-privacy-dp-sgd
```

Or with Docker Compose:

```bash
AUDIT_SECRET_KEY=your-secret docker compose up
```

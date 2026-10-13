# Differential Privacy and DP-SGD

Reference Python utilities for Gaussian and Laplace mechanisms, privacy-budget composition, gradient clipping with additive noise, and Gaussian Rényi differential privacy (RDP) accounting. A separate FastAPI service provides a **generic threshold-audit demonstration**, not clinical decision support or a DP-trained model.

## Capabilities

- **Noise mechanisms:** Scalar and vector Gaussian/Laplace noise, with sensitivity and privacy-parameter validation. Query responses contain the noisy value and mechanism parameters, **not** the original input or realized noise.
- **Sensitivity:** Count and histogram (add/remove adjacency), bounded sum (add/remove adjacency), and fixed-size bounded mean (replace-one adjacency).
- **Composition:** Basic and advanced `(ε,δ)` bounds and Gaussian RDP composition/conversion.
- **DP-SGD step:** Per-example L2 clipping, arithmetic mean, and Gaussian noise. The noise multiplier is dimensionless; for a batch of size `n` and clipping norm `C`, the implementation adds noise with standard deviation `2 × multiplier × C / n`, suitable for **fixed-size replace-one adjacency**. The engine **does not** implement data sampling, model parameter updates, or privacy amplification by subsampling.
- **Accountant:** Full-batch Gaussian RDP bound accumulated over steps; no subsampling amplification. Supply the Gaussian *noise multiplier*, not absolute standard deviation.
- **Web calculator:** Browser-side execution of the Python numerical engine through Pyodide, including Gaussian/Laplace queries, composition, gradient aggregation, and RDP accounting.
- **Local audit service:** An unrelated demonstration of numeric threshold alerts and an in-memory HMAC-chained audit log. Its threshold values are examples, **not validated clinical or privacy-conformance criteria**.

## Python installation and command line

Requires Python 3.10–3.12 in CI (package metadata permits Python 3.9+).

```bash
python -m pip install -e ".[dev]"
python -m pytest -q
python cli.py gaussian --value 100 --sensitivity 1 --epsilon 1 --delta 1e-5
python cli.py laplace --value 50 --sensitivity 1 --epsilon 1
python cli.py sensitivity --query mean --lower 10 --upper 20 --n 5
python cli.py dp-sgd --clip-norm 1 --epsilon 1 --delta 1e-5 --batch-size 32 --dim 10
python cli.py accountant --sigma 1.5 --steps 100 --delta 1e-5
```

Other calculator commands: `noise-scale`, `composition`, and `budget`. Run `python cli.py --help` for arguments. A separate installed command `differential-privacy-dp-sgd` runs the demonstration audit CLI; use `python -m diff_privacy_engine.cli --help` to explore it. The audit and calculator CLIs are intentionally separate.

## Browser application

The static source is in `docs/index.html`; GitHub Actions stages it beside the **authoritative** `diff_privacy_engine/engine.py` for Pages deployment. No backend, API keys, or Python server is required on the published page. The runtime downloads Pyodide from jsDelivr. Inputs and numerical outputs remain in the browser; do not use this reference calculator as a verified production privacy system.

To preview locally:

```bash
mkdir -p /tmp/dp-sgd-site
cp docs/index.html /tmp/dp-sgd-site/index.html
cp diff_privacy_engine/engine.py /tmp/dp-sgd-site/engine.py
python -m http.server 8080 --directory /tmp/dp-sgd-site
```

Visit `http://localhost:8080` with a current browser and an internet connection for the initial Pyodide download. GitHub Pages requires Actions-based Pages deployment to be enabled in repository settings. A live URL should only be advertised after a successful deployment and smoke test.

## Local REST API and Docker

```bash
python -m pip install -e ".[web]"
uvicorn agents.api:app --host 127.0.0.1 --port 8000
```

Visit `http://127.0.0.1:8000/` for the local API-backed threshold-audit console, or `/docs` for FastAPI OpenAPI documentation. Docker uses the same API:

```bash
docker build -t dp-sgd .
docker run --rm -p 127.0.0.1:8000:8000 -e AUDIT_SECRET_KEY="<random-secret>" dp-sgd
```

Set `AUDIT_SECRET_KEY` in the environment before `docker compose up`; Compose rejects missing keys. The audit-log endpoint `GET /api/audit/logs` is disabled unless `AUDIT_API_TOKEN` is set; when enabled, it requires `Authorization: Bearer <token>`.

**Security and privacy limitations:** The heuristic PHI scanner does not establish HIPAA de-identification. The FastAPI audit/chat endpoints have no authentication, so bind them to a trusted local interface and do not expose them to public networks. The "LLM" interface is a local mock: no external models or clinical validation are implemented. Browser/Python floating-point noise generation, composition assumptions, sensitivity bounds and threat models require independent assessment for sensitive or regulated datasets. CLI arguments can appear in shell history. The in-memory audit trail is not persistent storage.

## Development and verification

`pytest` exercises the numerical engine, both audit systems, and API. GitHub Actions installs the package and runs syntax checks, tests, and smoke commands against Python 3.10, 3.11, and 3.12. The Pages workflow publishes the browser calculator if Pages is enabled.

No production database or external service is required for the core numerical engine. MIT license: see [LICENSE](LICENSE).

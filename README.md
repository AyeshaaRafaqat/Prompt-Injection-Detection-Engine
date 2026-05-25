# Prompt Injection Detection Engine (PIDE)

A multi‑layer detection system for prompt‑injection attacks against large language models. It provides a fast, rule‑based Layer 1, embedding similarity, heuristic analysis and risk scoring.

---

## Quick Start (no extra files needed)

### 1. Clone the repository
```bash
git clone https://github.com/AyeshaaRafaqt/Prompt-Injection-Detection-Engine.git
cd Prompt-Injection-Detection-Engine
```

### 2. Choose how to run it
#### a) **Using the bundled Makefile** (recommended for beginners)
```bash
# Create virtual environment and install dependencies
make install

# Start the API server (FastAPI) – will be reachable at http://localhost:8000
make run-api

# In another terminal, launch the Gradio playground UI – http://localhost:7860
make run-ui

# Run the full test suite
make test
```

#### b) **Manual commands** (if you prefer the raw steps)
```bash
# Windows PowerShell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download en_core_web_sm
python scripts/build_exemplars.py

# API
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Gradio UI
python -m demo.gradio_app

# Run Tests
pytest tests/ -v
```

#### c) **Docker** (one‑liner for any platform with Docker installed)
```bash
# Build the image (once)
docker build -t pide .

# Run the container – exposes API on 8000 and UI on 7860
docker run -p 8000:8000 -p 7860:7860 pide
```

---

## Project Layout (for reference)
```
.
├── api/                # FastAPI REST gateway
├── config/             # YAML patterns & scoring config
├── data/               # Exemplars & FAISS index
├── demo/               # Gradio UI & LLM client helper
├── evaluation/         # Ablation & benchmark scripts
├── layers/            # Detection layers (L1‑L4)
├── logs/               # Audit logs (privacy‑preserving)
├── scripts/            # Utility scripts (e.g., build_exemplars.py)
├── tests/              # Pytest suite (39 tests)
├── Dockerfile          # Container build file
├── Makefile            # Helper targets for common tasks
├── pipeline.py         # Orchestrates layers & fail‑secure logic
├── requirements.txt
└── README.md            # (this file)
```

---

## Dockerfile (included in repo)
The repository already contains a minimal `Dockerfile` that:
1. Uses a lightweight Python 3.11 base image.
2. Creates a virtual environment, installs all dependencies, and downloads the spaCy model.
3. Exposes ports **8000** (API) and **7860** (Gradio UI).
4. Starts the API server by default.  The UI can be accessed via the same container at `http://localhost:7860`.

---

## Makefile (included in repo)
```makefile
VENV=venv
PYTHON=$(VENV)/Scripts/python.exe

# Create virtual environment
venv:
	python -m venv $(VENV)

# Install dependencies and spaCy model
install: venv
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -r requirements.txt
	$(PYTHON) -m spacy download en_core_web_sm

# Run API server (FastAPI)
run-api:
	$(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# Run Gradio Playground UI
run-ui:
	$(PYTHON) -m demo.gradio_app

# Execute the full test suite
test:
	$(PYTHON) -m pytest -q
```
Run any target with `make <target>` (e.g., `make run-ui`).

---

## License
MIT – feel free to fork, extend, and use in your own projects.

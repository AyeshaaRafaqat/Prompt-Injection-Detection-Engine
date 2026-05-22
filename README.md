# PIDE: Prompt Injection Detection Engine

PIDE is a multi-layer Information Security university project designed to detect and mitigate Prompt Injection attacks against Large Language Models (LLMs). The engine follows the **Defence-in-Depth** principle, ensuring that if one security layer fails or is bypassed, subsequent layers provide overlapping protection. By implementing these layers, PIDE addresses the **Integrity** and **Availability** components of the CIA triad, preventing attackers from hijacking model instructions or extracting sensitive system prompts.

## Directory Tree
```text
.
├── api/                # FastAPI REST Gateway
├── config/             # YAML Configuration (Regex patterns, Scoring weights)
├── data/               # Semantic exemplars and FAISS index storage
├── demo/               # Gradio Interactive UI
├── evaluation/         # Performance metrics and ablation scripts
├── layers/             # Core detection layers (L1-L4)
├── logs/               # Audit and error logs (Privacy-preserving)
├── notebooks/          # Research and threshold tuning
├── scripts/            # Data preparation and utility scripts
├── tests/              # Pytest suite
├── pipeline.py         # Orchestration and fail-secure logic
├── requirements.txt    # Pinned dependencies
└── README.md           # Project documentation
```

## Setup Instructions

1. **Create Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

2. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download spaCy Model:**
   ```bash
   python -m spacy download en_core_web_sm
   ```

4. **Prepare Data:**
   ```bash
   python scripts/build_exemplars.py
   ```

## Running the Engine

- **API Server:**
   ```bash
   uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

- **Gradio Demo:**
   ```bash
   python demo/gradio_app.py
   ```

- **Run Tests:**
   ```bash
   pytest tests/ -v
   ```

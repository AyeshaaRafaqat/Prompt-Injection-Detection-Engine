# PIDE Makefile
# Simplifies installation, testing, and deployment.

.PHONY: install test evaluate ablation api demo build-data clean

install:
	python -m venv venv
	./venv/Scripts/pip install -r requirements.txt
	./venv/Scripts/python -m spacy download en_core_web_sm

test:
	./venv/Scripts/pytest tests/ -v --tb=short

evaluate:
	./venv/Scripts/python evaluation/evaluate.py

ablation:
	./venv/Scripts/python evaluation/ablation.py

api:
	./venv/Scripts/uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

demo:
	./venv/Scripts/python demo/gradio_app.py

build-data:
	./venv/Scripts/python scripts/build_exemplars.py

clean:
	rm -rf logs/ audit.jsonl data/attack_exemplars.json evaluation/results/
	find . -type d -name "__pycache__" -exec rm -rf {} +

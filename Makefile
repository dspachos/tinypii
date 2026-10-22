# tinypii pipeline — one make target per phase. Run everything from the repo root.

VENV        := .venv
PY          := $(VENV)/bin/python
VENV_EXPORT := .venv-export
PY_EXPORT   := $(VENV_EXPORT)/bin/python
MODEL_DIR   := models/flan-t5-small-pii

.PHONY: help install seed data smoke train evaluate onnx export serve test test-browser clean

help: ## list targets
	@awk 'BEGIN {FS = ":.*##"} /^[a-zA-Z0-9_-]+:.*##/ {printf "  %-14s %s\n", $$1, $$2}' $(MAKEFILE_LIST)

install: ## create both python virtualenvs + node dev dependencies
	python3 -m venv $(VENV) && $(VENV)/bin/pip install -r requirements.txt
	python3 -m venv $(VENV_EXPORT) && $(PY_EXPORT) -m pip install -r requirements-export.txt
	npm install

seed: ## phase 1 — free template pairs -> data/seed.jsonl
	$(PY) phase-1-generate-data/generate.py seed

data: ## phase 1 — teacher-LLM pairs -> data/synthetic.jsonl (needs .env.local, API cost)
	$(PY) phase-1-generate-data/generate.py teacher --target 8647

smoke: ## phase 2 — tiny training run on seed data, validates the loop
	$(PY) phase-2-train/finetune.py --smoke

train: ## phase 2 — full fine-tune -> models/flan-t5-small-pii
	$(PY) phase-2-train/finetune.py --epochs 5 --batch 8 --grad-accum 2

evaluate: ## phase 3 — acceptance gate on held-out split + adversarial set
	$(PY) phase-3-evaluate/evaluate.py

onnx: ## phase 4 — fp32 ONNX export -> models/onnx (runs in .venv-export)
	$(VENV_EXPORT)/bin/optimum-cli export onnx --model $(MODEL_DIR) --task text2text-generation-with-past models/onnx

export: ## phase 4 — INT8 quantize + assemble -> web/model (runs in .venv-export)
	$(PY_EXPORT) phase-4-export/export.py

serve: ## serve the browser client at http://127.0.0.1:8123
	python3 -m http.server 8123 --directory web

test: ## node smoke + parity tests (needs web/model and node_modules)
	node test/smoke.mjs && node test/parity.mjs

test-browser: ## headless browser test (run `make serve` in another terminal first)
	node test/browser-test.mjs

clean: ## remove generated artifacts (keeps virtualenvs and node_modules)
	rm -rf data models web/model trainer_checkpoints browser-test.png

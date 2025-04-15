# Byte-compiled / optimized / DLL files
__pycache__/
*.py[cod]
*$py.class

# C extensions
*.so

# Distribution / packaging
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
*.egg-info/
.installed.cfg
*.egg

# PyInstaller
*.manifest
*.spec

# Installer logs
pip-log.txt
pip-delete-this-directory.txt

# Unit test / coverage reports
htmlcov/
.tox/
.nox/
.coverage
.coverage.*
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.pytest_cache/

# Jupyter Notebook
.ipynb_checkpoints

# pyenv
.python-version

# mypy
.mypy_cache/
.dmypy.json
dmypy.json

# VSCode
.vscode/

# Environment
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/

# Mac
.DS_Store

# Data
data/# Advanced Multimodal Visual Question Answering (VQA) System

A modular, extensible, and production-ready system for answering complex questions about images, text, and speech using state-of-the-art Vision-Language Models (VLMs), Retrieval-Augmented Generation (RAG), and advanced data pipelines.

## Features

- Modular FastAPI backend with pluggable VLMs (Phi-4, LLaVA, etc.)
- Knowledge-augmented answering (RAG, SeBe-VQA, adaptive planning agent)
- Robust data pipelines (synthetic data, augmentation, text perturbation)
- Redis caching, configuration via YAML, and logging
- Continuous learning and feedback loop
- Placeholder AR frontend (React/Three.js)
- Dockerized deployment and monitoring (Triton, Prometheus, Grafana)
- Comprehensive test structure

## Installation

1. Clone the repo and `cd` into the project directory.
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ---
   
   # --- requirements.txt ---
   ```text
   fastapi
   uvicorn[standard]
   pydantic
   python-multipart
   Pillow
   torch
   torchvision
   torchaudio
   transformers
   accelerate
   sentencepiece
   albumentations
   opencv-python-headless
   redis
   PyYAML
   requests
   # For testing
   pytest
   httpx# scripts/__init__.py

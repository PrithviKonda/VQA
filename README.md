# Advanced Multimodal Visual Question Answering (VQA) System

## Overview

This repository implements a modular, extensible, and production-ready Multimodal VQA system, integrating advanced Visual-Language Models (VLMs), knowledge retrieval, and continuous learning.

## Directory Structure

- `src/`: Core backend code (API, VLM, data pipeline, knowledge, inference engine, cache, utils)
- `data/`: Raw, processed, and synthetic datasets
- `notebooks/`: Prototyping and experiments
- `scripts/`: Training, evaluation, and data generation scripts
- `deployment/`: Docker, orchestration, and monitoring configs
- `ar_frontend/`: Placeholder for AR-based frontend (Phase 6+)
- `tests/`: Unit and integration tests

## Quickstart

1. Install dependencies: `pip install -r requirements.txt`
2. Configure `config.yaml` as needed.
3. Run the API: `uvicorn src.api.main:app --reload`

## Phased Implementation

- **Phase 0:** Core foundation, API, config, logging, caching, Phi-4 VLM integration
- **Phase 1-7:** See `ProjectPlan.md` for details

## References

- See `Architecting an Advanced Multimodal Visual Question Answering System.md`
- See `componentandtechniques.md`
- See `ProjectPlan.md`

---
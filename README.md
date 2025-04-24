
# Multimodal Visual Question Answering (VQA) System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/) 

## Overview

This project implements an advanced, multimodal Visual Question Answering (VQA) system designed to answer complex questions based on visual, textual, and potentially speech inputs. It integrates Vision-Language Models (VLMs), sophisticated knowledge retrieval mechanisms (mRAG), adaptive reasoning agents, robust data pipelines, and Augmented Reality (AR) integration.

The system architecture emphasizes modularity and leverages techniques like contrastive alignment for retrieval (SeBe-VQA concept), adaptive planning (OmniSearch concept), multi-stage response generation, and continuous learning to provide accurate, reliable, and context-aware answers. It is designed to be adaptable for various domains, including healthcare and industrial inspection [cite: 267, 412-430].

## Features

* **Multimodal Input:** Accepts questions via text (and potentially speech via models like Phi-4 Multimodal). Processes complex visual inputs (images, diagrams).
* **Foundation VLMs:** Utilizes powerful VLMs like Phi-4 Multimodal and potentially LLaVA-NeXT for core visual and language understanding.
* **Advanced Knowledge Augmentation:** Implements multi-modal Retrieval-Augmented Generation (mRAG) to fetch external knowledge.
    * Includes concepts for specialized retrieval alignment (SeBe-VQA) and MLLM-based re-ranking.
    * Supports domain-specific knowledge connectors (e.g., PubMed).
* **Sophisticated Reasoning:** Incorporates concepts for an Adaptive Planning Agent (based on OmniSearch) to decompose complex questions and plan information retrieval steps.
* **Hybrid Response Generation:** Employs a multi-stage pipeline involving candidate generation, knowledge filtering/integration, and ranking (potentially using external judge models) [cite: 377-385].
* **Robust Data Pipeline:** Includes image/text preprocessing, advanced image augmentation (Albumentations), text perturbations, and structures for synthetic data generation (CoSyn, BiomedCLIP+LLM concepts) [cite: 344-352].
* **Scalable Deployment:** Designed for containerized deployment using Docker, with high-performance inference serving via NVIDIA Triton and API served by FastAPI.
* **Monitoring:** Integrates with Prometheus and Grafana for performance monitoring [cite: 407-408, 411].
* **Continuous Learning:** Includes structures for active learning (uncertainty sampling) and user feedback integration to enable ongoing improvement.
* **Augmented Reality Frontend:** Features a prototype web-based AR frontend (React Three Fiber) to visualize VQA results in context.

## Project Structure

The project follows a modular structure to separate concerns:

```
.
├── .gitignore
├── README.md
├── requirements.txt
├── config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── synthetic/
│
├── notebooks/
│   ├── 01_data_exploration.ipynb # Placeholder content ok
│   ├── 02_model_testing.ipynb    # Placeholder content ok
│   └── 03_rag_experiments.ipynb  # Placeholder content ok
│
├── src/
│   ├── __init__.py
│   ├── api/
│   │   ├── __init__.py
│   │   ├── main.py           # Implement Phase 0 endpoint(s)
│   │   ├── models.py         # Implement Phase 0 models
│   │   └── dependencies.py   # Implement Phase 0 dependencies (e.g., get_redis_client, get_vlm_service)
│   ├── vlm/
│   │   ├── __init__.py
│   │   ├── inference.py      # Implement basic Phase 0 inference for Phi-4
│   │   ├── loading.py        # Implement Phase 0 loading for Phi-4
│   │   └── wrappers/
│   │       ├── __init__.py
│   │       ├── phi4.py       # Define structure
│   │       └── llava.py      # Placeholder structure for Phase 3
│   ├── data_pipeline/        # Create files/structure for Phase 1
│   │   ├── __init__.py
│   │   ├── datasets.py       # Placeholder structure
│   │   ├── preprocessing.py  # Placeholder structure
│   │   ├── augmentation.py   # Placeholder structure
│   │   ├── synthetic_data.py # Placeholder structure
│   │   └── text_perturb.py   # Placeholder structure
│   ├── knowledge/            # Create files/structure for Phase 2 & 3
│   │   ├── __init__.py
│   │   ├── retriever.py      # Placeholder structure
│   │   ├── mrag.py           # Placeholder structure
│   │   ├── sebe_vqa.py       # Placeholder structure
│   │   ├── planner.py        # Placeholder structure
│   │   └── connectors/
│   │       ├── __init__.py
│   │       ├── base_connector.py # Placeholder structure
│   │       └── pubmed.py       # Placeholder structure
│   ├── inference_engine/     # Create files/structure for Phase 3
│   │   ├── __init__.py
│   │   ├── response_generator.py # Placeholder structure
│   │   └── ranker.py             # Placeholder structure
│   ├── cache/
│   │   ├── __init__.py
│   │   └── redis_cache.py    # Implement Phase 0 caching functions
│   ├── continuous_learning/  # Create files/structure for Phase 6
│   │   ├── __init__.py
│   │   ├── active_learner.py # Placeholder structure
│   │   └── feedback.py       # Placeholder structure
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py # Implement Phase 0 logging setup
│       └── helpers.py        # Implement Phase 0 config loading
│
├── scripts/                  # Create files/structure for Phase 1 & 4
│   ├── __init__.py
│   ├── run_training.py           # Placeholder structure
│   ├── run_evaluation.py         # Placeholder structure
│   ├── generate_synthetic_data.py # Placeholder structure
│   └── db_manage.py              # Placeholder structure (if needed for vector DB later)
│
├── deployment/               # Create files/structure for Phase 5
│   ├── Dockerfile.api        # Placeholder content ok
│   ├── Dockerfile.worker     # Placeholder content ok
│   ├── docker-compose.yml    # Placeholder content ok
│   ├── triton_repo/
│   │   └── vqa_model/
│   │       ├── 1/
│   │       │   └── model.pt      # Placeholder
│   │       └── config.pbtxt    # Placeholder
│   └── monitoring/
│       ├── prometheus/
│       │   └── prometheus.yml  # Placeholder
│       └── grafana/
│           └── provisioning/
│               ├── dashboards/ # Placeholder
│               └── datasources/ # Placeholder
│
├── ar_frontend/              # Create files/structure for Phase 6
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── hooks/
│   │   ├── App.js            # Placeholder content ok
│   │   └── index.js          # Placeholder content ok
│   ├── package.json        # Basic placeholder
│   └── README.md             # Basic placeholder (to be filled later)
│
└── tests/                    # Create structure for Phase 0 & beyond
    ├── __init__.py
    ├── api/                  # Add basic Phase 0 tests
    ├── data_pipeline/
    ├── knowledge/
    ├── vlm/
    └── test_utils.py
```

## Technology Stack

* **Backend:** Python 3.10+, FastAPI, Uvicorn
* **ML/VLMs:** PyTorch, Hugging Face Transformers (Phi-4, LLaVA-NeXT concepts)
* **Data Handling:** Albumentations, OpenCV, Pillow, PyYAML
* **Knowledge Retrieval:** Sentence Transformers, FAISS/ChromaDB (conceptual), Requests (for connectors)
* **Caching:** Redis, aioredis
* **Deployment:** Docker, Docker Compose, NVIDIA Triton Inference Server
* **Monitoring:** Prometheus, Grafana
* **Frontend:** React, React Three Fiber, Drei, XR Interaction Toolkit (@react-three/xr), JavaScript/TypeScript
* **Testing:** Pytest

## Setup / Installation

**1. Clone Repository:**

```bash
git clone <your-repository-url>
cd <repository-name>
```

**2. Set up Python Environment:**

It's recommended to use a virtual environment (like `venv` or `conda`).

```bash
# Using venv
python -m venv venv
source venv/bin/activate # On Windows use `venv\Scripts\activate`

# Using conda
conda create -n vqa_env python=3.10
conda activate vqa_env
```

**3. Install Dependencies:**

Install Python dependencies listed in `requirements.txt`. Note that dependencies are added incrementally based on the implementation phase. Ensure you have the latest requirements for the features you intend to use.

```bash
pip install -r requirements.txt
# Potentially install PyTorch with specific CUDA version if needed
# pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# Potentially install faiss-gpu if using GPU for retrieval
# pip install faiss-gpu # Or faiss-cpu
```

**4. Configuration:**

Copy the example configuration file and modify it as needed:

```bash
cp config.example.yaml config.yaml
# Edit config.yaml with your settings (model paths, API keys, Redis host/port, etc.)
```

**5. Vector Store Setup (for RAG - Phase 2 onwards):**

If using RAG features, you'll need to set up and populate your vector database (e.g., FAISS, ChromaDB) according to the implementation in `src/knowledge/retriever.py`. This might involve running a separate script to index your knowledge documents.

**6. Model Weights:**

Download the required VLM model weights (e.g., Phi-4 Multimodal) from Hugging Face or the specified source. Ensure the paths in `config.yaml` point to the correct locations.

**7. Frontend Setup (Phase 6 onwards):**

```bash
cd ar_frontend
npm install # or yarn install
cd ..
```

## Configuration

Key configurations are managed in `config.yaml`. This includes:

* `vlm_settings`: Model identifiers (e.g., Hugging Face IDs), paths to local weights.
* `redis_settings`: Host, port, database number for Redis cache.
* `retriever_settings`: Model name for sentence transformer, path to vector index.
* `api_settings`: Host, port for the FastAPI server.
* `training_args`: Hyperparameters for fine-tuning (learning rate, batch size, etc.).
* `evaluation_settings`: Paths to evaluation datasets, metrics to compute.
* `triton_settings`: URL for the Triton Inference Server (used in Phase 5+).
* (Potentially API keys for external services like PubMed connector or judge models).

Load configuration using the helper function in `src/utils/helpers.py`.

## Usage

**1. Running the API Server:**

* **Directly using Uvicorn (for development):**
    ```bash
    # Ensure environment is activated and you are in the project root
    uvicorn src.api.main:app --host 0.0.0.0 --port 8000 --reload
    ```
    Access the interactive documentation at `http://localhost:8000/docs`.

* **Using Docker Compose (Phase 5 onwards):**
    This will start the API, Redis, and potentially Triton.
    ```bash
    # Ensure Docker Desktop is running
    docker-compose up --build
    ```
    The API should be accessible at `http://localhost:8000` (or the port mapped in `docker-compose.yml`).

**2. Making API Requests:**

The primary endpoint is `/vqa`. You can send requests using `curl` or any HTTP client.

* **Basic Request (Image + Question):**
    ```bash
    curl -X POST "http://localhost:8000/vqa" \
         -H "Content-Type: multipart/form-data" \
         -F "image=@/path/to/your/image.jpg" \
         -F "request_data='{\"question\": \"What color is the object?\", \"use_rag\": false}'"
         # request_data is a JSON string containing the question and other parameters
    ```
* **Request with RAG (Phase 2 onwards):**
    ```bash
    curl -X POST "http://localhost:8000/vqa" \
         -H "Content-Type: multipart/form-data" \
         -F "image=@/path/to/your/image.jpg" \
         -F "request_data='{\"question\": \"What is known about this landmark?\", \"use_rag\": true}'"
    ```

**3. Running Training (Phase 4 onwards):**

Use the `run_training.py` script. Adjust parameters in `config.yaml` or via command-line arguments (if implemented).

```bash
# Ensure environment is activated
python scripts/run_training.py --config_path config.yaml --output_dir ./models/fine_tuned_vlm
```

**4. Running Evaluation (Phase 4 onwards):**

Use the `run_evaluation.py` script.

```bash
# Ensure environment is activated
python scripts/run_evaluation.py --config_path config.yaml --model_path ./models/fine_tuned_vlm --dataset_path /path/to/eval_data --output_file ./evaluation_results.json
```

**5. Running Synthetic Data Generation (Phase 1 onwards):**

Use the `generate_synthetic_data.py` script (note: initial implementation might be placeholders).

```bash
# Ensure environment is activated
python scripts/generate_synthetic_data.py --config_path config.yaml --output_dir ./data/synthetic
```

**6. Running Tests (Phase 7 onwards):**

Use `pytest`.

```bash
# Ensure environment is activated and testing dependencies are installed
pytest tests/
# For coverage report (requires pytest-cov)
pytest --cov=src tests/
```

**7. Using the AR Frontend (Phase 6 onwards):**

* Start the backend API server (using Uvicorn or Docker Compose).
* Start the React development server:
    ```bash
    cd ar_frontend
    npm start # or yarn start
    ```
* Access the frontend URL (usually `http://localhost:3000`) on an AR-compatible device/browser. Use the UI to interact with the VQA system.

## Deployment (Phase 5 onwards)

The system is designed for deployment using Docker and NVIDIA Triton Inference Server.

1.  **Build Docker Images:** Use the provided Dockerfiles (`deployment/Dockerfile.api`, `deployment/Dockerfile.worker`).
2.  **Prepare Triton Model Repository:** Export the trained VLM(s) to a format compatible with Triton (e.g., TorchScript, ONNX) and place them in the `deployment/triton_repo/vqa_model/1/` directory. Configure `deployment/triton_repo/vqa_model/config.pbtxt` accordingly.
3.  **Run Services:** Use `docker-compose.yml` as a template or deploy containers using orchestration tools like Kubernetes. Ensure the API container can reach the Triton server and Redis instance.

## Monitoring (Phase 5 onwards)

Basic monitoring is set up using Prometheus and Grafana [cite: 407-408, 411].

* Prometheus scrapes metrics from Triton and potentially the FastAPI application.
* Grafana uses Prometheus as a data source to visualize key performance indicators (latency, throughput, resource utilization, cache hit rate). Configure dashboards in Grafana based on collected metrics.

## Continuous Learning (Phase 6 onwards)

The system includes components for continuous improvement [cite: 431-445]:

* **Active Learning:** Identifies uncertain predictions for review and targeted relabeling.
* **User Feedback:** An API endpoint (`/feedback`) allows collecting explicit user ratings and comments.
* This data can be incorporated into subsequent retraining cycles (Phase 4 scripts need adaptation) to improve model performance over time.

## Contributing

Contributions are welcome! Please follow standard coding practices (PEP 8, typing, docstrings). Open an issue to discuss major changes before submitting a pull request. (Add more specific guidelines as needed).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details (assuming MIT, create a LICENSE file).

## Acknowledgements

* Mention any libraries, datasets, or papers that heavily influenced the project.
* Reference the core architecture document [cite: 1-528].

## Contact

* Your Name / Team Name - contact@example.com
* Project Link: <your-repository-url>

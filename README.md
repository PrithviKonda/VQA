# Advanced Multimodal VQA System

This project implements a modular, scalable Visual Question Answering (VQA) system integrating state-of-the-art Vision-Language Models (VLMs), knowledge retrieval, synthetic data pipelines, and AR visualization.

## Setup
1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Configure `config.yaml` as needed.
3. Run the backend:
   ```bash
   uvicorn src.api.main:app --reload
   ```

## Project Structure
See the tree in the project root for an overview of folders and files.

## Usage
- Use the FastAPI docs at `/docs` for API exploration.
- Place data in the `data/` directory.
- See `notebooks/` for experiments and visualization.
- AR frontend is in `ar_frontend/`.

## Contributing
Pull requests and issues are welcome!

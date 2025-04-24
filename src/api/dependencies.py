# src/api/dependencies.py
"""
FastAPI dependency injection functions (e.g., DB session, config, cache).
"""
from src.utils.config_loader import load_config

import os
from tritonclient.http import InferenceServerClient, InferInput

TRITON_SERVER_URL = os.getenv("TRITON_SERVER_URL", "http://localhost:8000")
TRITON_MODEL_NAME = "vqa_model"

def triton_infer(image_tensor, question_tensor):
    client = InferenceServerClient(url=TRITON_SERVER_URL)

    # Prepare inputs
    image_input = InferInput("image", image_tensor.shape, "FP32")
    image_input.set_data_from_numpy(image_tensor)

    question_input = InferInput("question", question_tensor.shape, "INT32")
    question_input.set_data_from_numpy(question_tensor)

    # Run inference
    results = client.infer(
        model_name=TRITON_MODEL_NAME,
        inputs=[image_input, question_input]
    )
    answer = results.as_numpy("answer")
    return answer

def get_config():
    return load_config()

#!/usr/bin/env python3
"""
Evaluation script for the VQA system (Phi-4 Multimodal).
Usage:
    python scripts/run_evaluation.py --config config.yaml --model_path ./checkpoints --split val --output results.json
"""

import argparse
import json
import yaml
from typing import Dict, Any

from src.data_pipeline.datasets import get_vqa_dataset  # Assumed implemented
from src.vlm.loading import load_vlm_model, model, processor
from src.vlm.inference import perform_vqa
# from src.inference_engine.response_generator import ResponseGenerator  # For end-to-end eval (optional)
import evaluate

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Evaluate VQA VLM (Phi-4 Multimodal)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--model_path", type=str, required=True, help="Directory or file of trained model")
    parser.add_argument("--split", type=str, default="val", help="Dataset split to evaluate on")
    parser.add_argument("--output", type=str, required=True, help="File to save evaluation results (JSON)")
    args = parser.parse_args()

    config = load_config(args.config)
    eval_args = config.get("evaluation", {})
    dataset = get_vqa_dataset(split=args.split, config=config)

    # Load model and processor
    print("Loading model and processor...")
    load_vlm_model()  # Assumes model_path is set in config['vlm']['model_id']
    assert model is not None and processor is not None, "Model or processor not loaded."

    # Metric
    vqa_accuracy = evaluate.load("accuracy")

    results = []
    correct = 0
    total = 0

    print("Running evaluation...")
    for example in dataset:
        image = example['image']  # Assumed PIL.Image
        question = example['question']
        gt_answer = example['answer']

        pred_answer = perform_vqa(image, question)
        results.append({
            "question": question,
            "gt_answer": gt_answer,
            "pred_answer": pred_answer
        })

        # Simple accuracy: exact match
        acc = int(str(pred_answer).strip().lower() == str(gt_answer).strip().lower())
        vqa_accuracy.add(prediction=pred_answer, reference=gt_answer)
        correct += acc
        total += 1

    # Aggregate metrics
    accuracy = vqa_accuracy.compute()['accuracy'] if total > 0 else 0.0
    print(f"Evaluation complete. Accuracy: {accuracy:.4f} ({correct}/{total})")

    # Save results
    with open(args.output, 'w') as f:
        json.dump({
            "accuracy": accuracy,
            "results": results
        }, f, indent=2)
    print(f"Results saved to {args.output}")

if __name__ == '__main__':
    main()

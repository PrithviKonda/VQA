#!/usr/bin/env python3
"""
Training script for fine-tuning the VQA system (Phi-4 Multimodal).
Usage:
    python scripts/run_training.py --config config.yaml --output_dir ./checkpoints --epochs 3 --batch_size 8
"""

import argparse
import os
import yaml
from typing import Dict, Any

import torch
from transformers import TrainingArguments, Trainer
from src.vlm.loading import load_vlm_model, model, processor
from src.data_pipeline.datasets import get_vqa_dataset  # Assumed implemented
# from src.knowledge.sebe_vqa import ...  # Placeholder for SeBe-VQA alignment model training

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description="Fine-tune VQA VLM (Phi-4 Multimodal)")
    parser.add_argument("--config", type=str, required=True, help="Path to config.yaml")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save model checkpoints")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--learning_rate", type=float, default=None, help="Learning rate (overrides config)")
    args = parser.parse_args()

    config = load_config(args.config)
    train_args = config.get("training", {})
    epochs = args.epochs or train_args.get("epochs", 3)
    batch_size = args.batch_size or train_args.get("batch_size", 8)
    lr = args.learning_rate or train_args.get("learning_rate", 5e-5)

    # Load dataset
    print("Loading training dataset...")
    train_dataset = get_vqa_dataset(split="train", config=config)

    # Load model and processor
    print("Loading model and processor...")
    model_obj, processor_obj = load_vlm_model()
    assert model_obj is not None and processor_obj is not None, "Model or processor not loaded."

    # Prepare TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        learning_rate=lr,
        logging_steps=50,
        save_total_limit=2,
        report_to="none",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
    )

    # Data collator (placeholder: assumes dataset returns dict with 'input_ids', 'labels', etc.)
    def data_collator(features):
        return {k: torch.stack([f[k] for f in features]) for k in features[0]}

    # Trainer
    trainer = Trainer(
        model=model_obj,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=processor_obj,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete. Saving final model...")
    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

    # Placeholder for SeBe-VQA alignment model training
    # print("Training SeBe-VQA contrastive alignment model (not implemented)...")

if __name__ == '__main__':
    main()

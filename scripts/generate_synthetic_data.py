"""
Script to generate synthetic VQA data using pipeline stubs.

Author: VQA System Architect
"""

import argparse
from src.data_pipeline.synthetic_data import (
    generate_cosyn_synthetic_sample,
    generate_biomedclip_vqa_stub
)

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic VQA data")
    parser.add_argument("--concept", choices=["cosyn", "biomedclip"], required=True)
    parser.add_argument("--prompt", type=str, default="Draw a red square.")
    parser.add_argument("--seed_text", type=str, default="Chest X-ray")
    args = parser.parse_args()

    if args.concept == "cosyn":
        sample = generate_cosyn_synthetic_sample(args.prompt)
        print(f"CoSyn Sample: {sample}")
    elif args.concept == "biomedclip":
        sample = generate_biomedclip_vqa_stub(args.seed_text)
        print(f"BiomedCLIP Sample: {sample}")

if __name__ == "__main__":
    main()
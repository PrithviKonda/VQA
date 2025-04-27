# scripts/generate_synthetic_data.py
"""
Runner script for synthetic data generation placeholder.
"""
import argparse
import logging
from src.data_pipeline.synthetic_data import SyntheticDataGenerator

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic VQA data.")
    parser.add_argument("--concepts", nargs="+", required=True, help="List of concepts to generate data for.")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider: openai or anthropic.")
    parser.add_argument("--output", type=str, required=True, help="Output file path (JSONL).")
    parser.add_argument("--mode", type=str, default="cosyn", choices=["cosyn", "biomedclip"], help="Generation mode.")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    generator = SyntheticDataGenerator(provider=args.provider)

    with open(args.output, "w") as f:
        for concept in args.concepts:
            try:
                if args.mode == "cosyn":
                    sample = generator.generate_cosyn_sample(concept)
                else:
                    sample = generator.generate_biomedclip_sample(concept)
                f.write(f"{sample}\n")
                logging.info(f"Generated sample for concept: {concept}")
            except Exception as e:
                logging.error(f"Failed to generate sample for {concept}: {e}")

if __name__ == "__main__":
    main()

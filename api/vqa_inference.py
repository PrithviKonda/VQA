import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models.llava_vlm import LlavaVLM

def main():
    if len(sys.argv) < 3:
        print("Usage: python api/vqa_inference.py <image_path> <question>")
        return
    image_path = sys.argv[1]
    question = sys.argv[2]
    model = LlavaVLM()
    answer = model.predict(image_path, question)
    print(f"Q: {question}\nA: {answer}")

if __name__ == "__main__":
    main()

from transformers import AutoProcessor, AutoModelForVision2Seq

model_id = "Salesforce/blip2-opt-2.7b"

try:
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
    model = AutoModelForVision2Seq.from_pretrained(model_id, trust_remote_code=True)
    print("BLIP-2 loaded successfully.")
except Exception as e:
    print(f"BLIP-2 load failed: {e}")

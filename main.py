# main.py
# Basic VQA System using Phi-4 Multimodal and FastAPI

# --- Dependencies ---
# Install necessary libraries:
# pip install fastapi uvicorn[standard] python-multipart Pillow torch torchvision torchaudio transformers accelerate sentencepiece requests

import io
import torch
from PIL import Image
import requests # Only needed if loading images from URLs for testing
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from transformers import (
    Phi4MultimodalProcessor,
    Phi4MultimodalForConditionalGeneration,
    AutoProcessor # Can potentially use AutoProcessor as well
)

# --- Global Variables ---
# These will hold the loaded model and processor to avoid reloading on every request
model = None
processor = None
device = "cuda" if torch.cuda.is_available() else "cpu"
# Use the model ID suggested in the research document
MODEL_ID = "microsoft/Phi-4-multimodal-instruct"

# --- VLM Integration Logic ---

def load_vlm_model():
    """
    Loads the Phi-4 Multimodal model and processor.
    Handles model placement on CPU or GPU.
    """
    global model, processor
    if model is not None and processor is not None:
        print("Model and processor already loaded.")
        return

    print(f"Attempting to load model '{MODEL_ID}' on device: {device}...")
    try:
        # Determine torch dtype based on device for efficiency
        torch_dtype = torch.float16 if device == "cuda" else torch.float32

        # Load the processor associated with the model
        # trust_remote_code=True is required for Phi-4 Multimodal
        processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
        print("Processor loaded.")

        # Load the model
        model = Phi4MultimodalForConditionalGeneration.from_pretrained(
            MODEL_ID,
            trust_remote_code=True,
            torch_dtype=torch_dtype,
            # device_map="auto" # Can use accelerate's device_map for automatic distribution
            # Or explicitly move to device:
        ).to(device)

        # Set model to evaluation mode
        model.eval()

        print(f"Model '{MODEL_ID}' loaded successfully on {device}.")

    except ImportError as e:
        print(f"ImportError loading model: {e}. Make sure 'transformers', 'torch', 'accelerate' are installed.")
        raise HTTPException(status_code=500, detail=f"Server configuration error: Missing libraries. {e}")
    except Exception as e:
        print(f"An unexpected error occurred while loading the model: {e}")
        # Depending on desired behavior, could raise HTTPException or allow app to run degraded
        raise HTTPException(status_code=500, detail=f"Failed to load VLM model: {e}")


def perform_vqa(image: Image.Image, question: str) -> str:
    """
    Performs Visual Question Answering using the loaded Phi-4 model.

    Args:
        image: A PIL Image object.
        question: The text question string.

    Returns:
        The generated answer string.
    """
    if model is None or processor is None:
        # This case should ideally be prevented by the startup event,
        # but added as a safeguard.
        print("Error: VLM Model or processor not loaded.")
        raise HTTPException(status_code=503, detail="VLM Model not available.")

    try:
        # Format the prompt using the chat template expected by Phi-4 Multimodal.
        # The processor often handles this if the template is configured.
        # Example format (check model card for specifics):
        # <|user|>
        # <|image_1|>
        # {question}<|end|>
        # <|assistant|>
        # Let's construct the prompt manually to be sure.
        # Note: Using the processor's chat template might be more robust if available/configured.
        prompt = f"<|user|>\n<|image_1|>\n{question}<|end|>\n<|assistant|>"

        # Process the inputs (image + text prompt)
        # The processor prepares the image and tokenizes the text.
        inputs = processor(text=prompt, images=[image], return_tensors="pt").to(device)

        # Set generation parameters
        # These might need tuning based on desired output length and style
        generation_args = {
            "max_new_tokens": 100, # Limit response length
            "temperature": 0.7,    # Controls randomness (higher = more random)
            "do_sample": True,     # Enable sampling for more diverse answers
            "eos_token_id": processor.tokenizer.eos_token_id
        }

        print("Generating VQA response...")
        # Generate the response using the model
        with torch.no_grad(): # Disable gradient calculations for inference
            generate_ids = model.generate(**inputs, **generation_args)

        # Decode the generated token IDs back into text
        # The generated IDs include the input prompt, so slice to get only the new tokens.
        input_token_len = inputs["input_ids"].shape[1]
        response_ids = generate_ids[:, input_token_len:]
        response = processor.batch_decode(response_ids, skip_special_tokens=True)[0].strip()
        print(f"Raw response: {response}")

        return response

    except Exception as e:
        print(f"Error during VQA inference: {e}")
        # Provide a user-friendly error message
        raise HTTPException(status_code=500, detail=f"Error during VQA processing: {e}")


# --- FastAPI Application ---

app = FastAPI(
    title="Advanced Multimodal VQA System API",
    description="API endpoint for the VQA system based on Phi-4 Multimodal.",
    version="0.1.0"
)

@app.on_event("startup")
async def startup_event():
    """
    Load the VLM model when the FastAPI application starts.
    """
    print("Application startup: Initializing VLM...")
    load_vlm_model() # Load the model and processor into global variables

@app.get("/", summary="Root endpoint", description="Simple health check endpoint.")
async def read_root():
    """Returns a simple greeting message."""
    return {"message": "VQA System API is running."}

@app.post("/vqa/", summary="Perform Visual Question Answering")
async def run_vqa_endpoint(
    question: str = Form(..., description="The question to ask about the image."),
    image_file: UploadFile = File(..., description="The image file to analyze.")
):
    """
    Accepts an image file and a text question via form data.

    Processes the inputs using the Phi-4 Multimodal model and returns the answer.
    """
    print(f"Received VQA request. Question: '{question}', Image: '{image_file.filename}'")

    # --- Input Validation ---
    # Check image file type
    if not image_file.content_type.startswith("image/"):
        print(f"Invalid file type received: {image_file.content_type}")
        raise HTTPException(status_code=400, detail=f"Invalid file type '{image_file.content_type}'. Please upload an image (e.g., JPEG, PNG).")

    # --- Image Processing ---
    try:
        # Read image content from the uploaded file
        content = await image_file.read()
        # Open the image using Pillow
        image = Image.open(io.BytesIO(content))

        # Ensure image is in RGB format (common requirement for vision models)
        if image.mode != 'RGB':
            print(f"Converting image from {image.mode} to RGB.")
            image = image.convert('RGB')

        print(f"Image '{image_file.filename}' loaded successfully ({image.size[0]}x{image.size[1]}).")

    except Exception as e:
        print(f"Error reading or processing image file: {e}")
        raise HTTPException(status_code=400, detail=f"Could not read or process image file: {e}")

    # --- Perform VQA ---
    try:
        answer = perform_vqa(image, question)
        print(f"VQA successful. Answer: '{answer}'")

        # Return the result in a JSON response
        return JSONResponse(content={
            "question": question,
            "filename": image_file.filename,
            "answer": answer
        })
    except HTTPException as http_exc:
        # Re-raise exceptions that are already HTTPException (e.g., from perform_vqa)
        raise http_exc
    except Exception as e:
        # Catch any other unexpected errors during VQA
        print(f"Unexpected error during VQA processing for question '{question}': {e}")
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {e}")

# --- How to Run ---
# 1. Save this code as `main.py`.
# 2. Create a `requirements.txt` file with the following content:
#    ```
#    fastapi
#    uvicorn[standard]
#    python-multipart
#    Pillow
#    torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121) # Adjust CUDA version or use CPU version if needed
#    torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
#    torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
#    transformers
#    accelerate
#    sentencepiece
#    requests
#    ```
# 3. Install dependencies: `pip install -r requirements.txt`
# 4. Run the FastAPI server: `uvicorn main:app --reload --host 0.0.0.0 --port 8000`
#    (The `--reload` flag automatically restarts the server when code changes are detected)

# --- How to Test ---
# You can use tools like `curl` or Python's `requests` library, or the built-in Swagger UI.
#
# Using Swagger UI:
# - Navigate to http://localhost:8000/docs in your browser after starting the server.
# - Find the `/vqa/` endpoint, click "Try it out".
# - Enter your question in the `question` field.
# - Click the "Choose File" button for `image_file` and select an image.
# - Click "Execute".
#
# Using curl (example):
# curl -X POST "http://localhost:8000/vqa/" -H "accept: application/json" -H "Content-Type: multipart/form-data" -F "question=What color is the object?" -F "image_file=@/path/to/your/image.jpg"

```text
# requirements.txt

fastapi
uvicorn[standard]
python-multipart
Pillow
# --- PyTorch Installation ---
# Choose the command matching your system from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/)
# Example for CUDA 12.1:
torch --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
torchvision --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
torchaudio --index-url [https://download.pytorch.org/whl/cu121](https://download.pytorch.org/whl/cu121)
# Example for CPU only:
# torch
# torchvision
# torchaudio

# --- Hugging Face Transformers and dependencies ---
transformers
accelerate # Often helps with model loading and performance
sentencepiece # Often required by tokenizers

# --- Other ---
requests # For potential URL fetching, good to have

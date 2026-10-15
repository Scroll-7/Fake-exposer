"""
image_api.py

Local AI-image detection service (port 8001).

Runs the Nonescape Mini v0 ONNX classifier to estimate the probability that an
image is AI-generated. This is the only local image model that contributes real
signal to the detector.

Note
----
This service used to also host a generic cat/dog/wild/human classifier (the
"animal model"). That model was removed in the evaluation-hardening pass:

  * It contributed no fake-news signal — it only emitted cosmetic flags such as
    "Detected a 'dog' with 92% confidence".
  * Its class order was hardcoded (`['cat','dog','humans','wild']`) while the
    trainer derived order from folder names, so training without a `humans`
    class silently misaligned every prediction.

Human presence is now established by the InsightFace service on port 8002,
which is the correct tool for that job.
"""

from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from nonescape_preprocess import preprocess_nonescape

app = FastAPI(title="AI Image Detection API")

# ── AI Detection Model (Nonescape mini v0) ──
ai_session = None
try:
    ai_session = ort.InferenceSession("nonescape-mini-v0.onnx")
    print("AI Detection ONNX model loaded successfully!")
except Exception as e:
    print(f"Error loading AI detection model: {e}")
    print("Run: npm run download-models")
    print("Or download manually from: https://github.com/e3ntity/nonescape")


def preprocess_ai_detection(image_bytes):
    """Nonescape preprocessing (shared with the eval harness)."""
    return preprocess_nonescape(image_bytes)


class AiDetectResponse(BaseModel):
    ai_probability: float
    authentic_probability: float


@app.get("/health")
def health():
    return {"status": "ok", "ai_detect_model_loaded": ai_session is not None}


@app.post("/detect_ai", response_model=AiDetectResponse)
async def detect_ai(file: UploadFile = File(...)):
    if not ai_session:
        return {"error": "AI detection model not loaded. Ensure nonescape-mini-v0.onnx is in the directory."}
    image_bytes = await file.read()
    input_data = preprocess_ai_detection(image_bytes)
    input_name = ai_session.get_inputs()[0].name
    outputs = ai_session.run(None, {input_name: input_data})
    # Logits → softmax → [authentic, ai]
    logits = outputs[0][0]
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    return AiDetectResponse(
        ai_probability=float(probs[1]),
        authentic_probability=float(probs[0]),
    )


if __name__ == "__main__":
    import uvicorn
    # Run on port 8001 so it doesn't conflict with our text ML API
    uvicorn.run(app, host="127.0.0.1", port=8001)

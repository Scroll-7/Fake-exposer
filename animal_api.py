from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
import onnxruntime as ort
import numpy as np
from PIL import Image
import io

app = FastAPI(title="Animal Classification API")

# Load ONNX model
model_path = "animal_model.onnx"
session = None
try:
    session = ort.InferenceSession(model_path)
    print("ONNX Model loaded successfully!")
except Exception as e:
    print(f"Error loading ONNX model: {e}")

# Classes in alphabetical order — must match the torchvision ImageFolder convention
# After retraining with humans.zip: cat | dog | humans | wild
CLASSES = ['cat', 'dog', 'humans', 'wild']

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize((224, 224))
    
    # Convert to numpy array and scale to [0, 1] — explicitly float32
    img_data = np.array(image, dtype=np.float32) / 255.0
    
    # Normalize with ImageNet stats
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_data = (img_data - mean) / std
    
    # Transpose to Channel, Height, Width: (3, 224, 224)
    img_data = np.transpose(img_data, (2, 0, 1))
    
    # Add batch dimension: (1, 3, 224, 224)
    img_data = np.expand_dims(img_data, axis=0).astype(np.float32)
    
    return img_data

class PredictResponse(BaseModel):
    predicted_class: str
    confidence: float
    is_human: bool

@app.post("/predict_animal", response_model=PredictResponse)
async def predict_animal(file: UploadFile = File(...)):
    if not session:
        return {"error": "Model not loaded. Ensure animal_model.onnx is in the directory."}
        
    image_bytes = await file.read()
    input_data = preprocess_image(image_bytes)
    
    # Run ONNX inference
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: input_data})
    
    # Output is a batch of logits, e.g., shape (1, 3)
    logits = outputs[0][0]
    
    # Softmax to get probabilities
    exp_logits = np.exp(logits - np.max(logits))
    probs = exp_logits / exp_logits.sum()
    
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASSES[predicted_idx]
    confidence = float(probs[predicted_idx])
    
    return PredictResponse(
        predicted_class=predicted_class,
        confidence=confidence,
        is_human=(predicted_class == 'humans'),
    )

if __name__ == "__main__":
    import uvicorn
    # Run on port 8001 so it doesn't conflict with our text ML API
    uvicorn.run(app, host="127.0.0.1", port=8001)

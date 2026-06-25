from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import sys

app = FastAPI(title="Fake News Detection API")

# Load model and vectorizer
try:
    with open("fake_news_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("fake_news_vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)
    print("Model and vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
    # Don't exit here so the server can at least start and return errors if model is missing
    model = None
    vectorizer = None

class PredictRequest(BaseModel):
    text: str

class PredictResponse(BaseModel):
    is_fake: bool
    fake_probability: float
    real_probability: float

@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model or not vectorizer:
        return {"error": "Model not loaded. Please train the model first."}
        
    text = request.text
    # Transform text
    X = vectorizer.transform([text])
    
    # Predict (assumes model classes are [0, 1])
    # For WELFake, typically 1 = Fake, 0 = Real
    probabilities = model.predict_proba(X)[0]
    
    # Depending on how the labels mapped. Usually proba[0] is class 0, proba[1] is class 1.
    # We will assume class 1 is Fake for this dataset.
    real_prob = float(probabilities[0])
    fake_prob = float(probabilities[1]) if len(probabilities) > 1 else 0.0
    
    return PredictResponse(
        is_fake=bool(fake_prob > 0.5),
        fake_probability=fake_prob,
        real_probability=real_prob
    )

if __name__ == "__main__":
    import uvicorn
    # Run the API on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)

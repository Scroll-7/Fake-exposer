import json
import pickle

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Fake News Detection API")

# Default assumption if no model card is present. The trainer writes a card
# (fake_news_model.card.json) recording the real/fake label indices, so the API
# never has to guess — a silent real/fake inversion was a real risk before.
LABEL_MAP = {"real": 0, "fake": 1}
CARD_PATH = "fake_news_model.card.json"

try:
    with open(CARD_PATH, encoding="utf-8") as f:
        LABEL_MAP = json.load(f).get("label_map", LABEL_MAP)
    print(f"Loaded model card label_map: {LABEL_MAP}")
except Exception:
    print(f"No model card ({CARD_PATH}); assuming label_map {LABEL_MAP}")

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


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None and vectorizer is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    if not model or not vectorizer:
        return {"error": "Model not loaded. Please train the model first."}

    X = vectorizer.transform([request.text])
    probabilities = model.predict_proba(X)[0]
    prob_by_class = dict(zip(list(model.classes_), probabilities))

    fake_prob = float(prob_by_class.get(LABEL_MAP["fake"], 0.0))
    real_prob = float(prob_by_class.get(LABEL_MAP["real"], 0.0))

    return PredictResponse(
        is_fake=bool(fake_prob > 0.5),
        fake_probability=fake_prob,
        real_probability=real_prob,
    )


if __name__ == "__main__":
    import uvicorn
    # Run the API on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)

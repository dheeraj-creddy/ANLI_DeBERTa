"""
FastAPI Inference API for ANLI Classification
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import List, Dict
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="ANLI Multi-Class Classifier",
    description="Natural Language Inference API using DeBERTa-v3",
    version="1.0.0"
)

# Global variables
#MODEL_PATH = "./models/anli_deberta_model"
MODEL_PATH = "./models/saved_model_debertav3_anli_r2_tpu"
MAX_LENGTH = 256
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LABEL_NAMES = {0: 'Entailment', 1: 'Neutral', 2: 'Contradiction'}

model = None
tokenizer = None


# Request/Response models
class PredictionRequest(BaseModel):
    premise: str = Field(..., example="A person is walking a dog in the park")
    hypothesis: str = Field(..., example="A person is outside")


class BatchPredictionRequest(BaseModel):
    pairs: List[Dict[str, str]] = Field(
        ...,
        example=[
            {"premise": "A dog runs", "hypothesis": "An animal moves"},
            {"premise": "It's raining", "hypothesis": "It's sunny"}
        ]
    )


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: Dict[str, float]


@app.on_event("startup")
async def load_model():
    """Load model on startup"""
    global model, tokenizer

    try:
        logger.info(f"Loading model from {MODEL_PATH}")
        logger.info(f"Using device: {DEVICE}")

        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()

        logger.info("✓ Model loaded successfully")
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        raise


def predict_nli(premise: str, hypothesis: str) -> Dict:
    """Make NLI prediction"""
    # Tokenize
    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True
    ).to(DEVICE)

    # Predict
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred_idx = torch.argmax(probs, dim=-1).item()
        confidence = probs[0][pred_idx].item()

    return {
        "prediction": LABEL_NAMES[pred_idx],
        "confidence": float(confidence),
        "probabilities": {
            LABEL_NAMES[i]: float(probs[0][i].item())
            for i in range(3)
        }
    }


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "service": "ANLI Multi-Class Classifier",
        "version": "1.0.0",
        "model": "DeBERTa-v3-base",
        "endpoints": {
            "/predict": "POST - Single prediction",
            "/batch_predict": "POST - Batch predictions",
            "/health": "GET - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": DEVICE
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make single prediction"""
    try:
        if not request.premise.strip() or not request.hypothesis.strip():
            raise HTTPException(400, "Premise and hypothesis cannot be empty")

        result = predict_nli(request.premise, request.hypothesis)
        logger.info(f"Prediction: {result['prediction']} ({result['confidence']:.3f})")

        return result

    except Exception as e:
        logger.error(f"Error: {e}")
        raise HTTPException(500, str(e))


@app.post("/batch_predict")
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions"""
    try:
        if len(request.pairs) > 100:
            raise HTTPException(400, "Batch size too large (max 100)")

        results = []
        for i, pair in enumerate(request.pairs):
            premise = pair.get("premise", "").strip()
            hypothesis = pair.get("hypothesis", "").strip()

            if not premise or not hypothesis:
                results.append({
                    "index": i,
                    "error": "Missing premise or hypothesis"
                })
                continue

            try:
                result = predict_nli(premise, hypothesis)
                results.append({"index": i, **result})
            except Exception as e:
                results.append({"index": i, "error": str(e)})

        return {"count": len(results), "results": results}

    except Exception as e:
        raise HTTPException(500, str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
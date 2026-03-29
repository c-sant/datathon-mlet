from fastapi import FastAPI
from pydantic import BaseModel

from app.core.config import settings

app = FastAPI(
    title=settings.app_name,
    version=settings.model_version,
)


class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    label: str
    score: float
    model_version: str


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "app_name": settings.app_name,
        "environment": settings.environment,
        "model_version": settings.model_version,
    }


@app.post("/predict", response_model=PredictResponse)
def predict(payload: PredictRequest) -> PredictResponse:
    score = min(len(payload.text) / 100, 0.99)
    label = "long_text" if len(payload.text) >= 30 else "short_text"

    return PredictResponse(
        label=label,
        score=round(score, 4),
        model_version=settings.model_version,
    )

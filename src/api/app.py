"""
Module: src/api/app.py
Role: Production FastAPI REST Gateway for Startup Market Analyzer.
"""

import os
from datetime import datetime, timezone
from typing import List, Optional, Dict, Any
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

from src.domain.models import CompanyProfile, ArticleRecord, ensure_utc
from src.features.fusion import FeatureFusionPipeline
from src.models.explainability import ModelExplainer
from src.models.stacking import OutOfFoldStackingClassifier
from src.nlp.sentiment_engine_ import HybridNLPEngine

# MUST BE DECLARED AT MODULE ROOT SO UVICORN FINDS "app"
app = FastAPI(
    title="Startup Market Analyzer (SMA) API",
    version="1.0.0",
    description="Automated Multimodal Due-Diligence & Viability Prediction Engine",
)

_pipeline_bundle = {}


def get_pipeline():
    if not _pipeline_bundle:
        artifact_path = "models/stacking_ensemble_v1.pkl"
        if not os.path.exists(artifact_path):
            from src.models.train_and_save import train_and_persist
            train_and_persist()

        model = OutOfFoldStackingClassifier()
        model.load_artifacts(artifact_path)
        nlp = HybridNLPEngine(use_transformer=False)
        fusion = FeatureFusionPipeline(nlp_engine=nlp)

        _pipeline_bundle["model"] = model
        _pipeline_bundle["fusion"] = fusion

    return _pipeline_bundle["model"], _pipeline_bundle["fusion"]


# ----------------- SCHEMAS -----------------
class ArticlePayload(BaseModel):
    article_id: str
    source: str
    title: str
    content: str
    published_at: datetime


class AnalyzeRequest(BaseModel):
    company_id: str = Field(..., example="spacex")
    company_name: str = Field(..., example="SpaceX")
    total_funding_usd_m: float = Field(..., ge=0.0, example=9800.0)
    employee_count: int = Field(..., ge=1, example=13000)
    company_age_years: int = Field(..., ge=0, example=22)
    growth_rate: float = Field(..., example=0.18)
    industry_sector: str = Field(default="Enterprise SaaS", example="Aerospace")
    as_of_date: Optional[datetime] = Field(default=None)
    articles: List[ArticlePayload] = Field(default_factory=list)


class PredictionResponse(BaseModel):
    company_id: str
    predicted_class: int
    predicted_label: str
    probabilities: Dict[str, float]
    fused_features: Dict[str, float]


class ExplainResponse(BaseModel):
    company_id: str
    predicted_class: int
    top_positive_drivers: Dict[str, float]
    top_negative_drags: Dict[str, float]


# ----------------- ROUTES -----------------
@app.get("/api/v1/healthz", status_code=status.HTTP_200_OK)
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "startup-market-analyzer-api",
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict_startup_growth(payload: AnalyzeRequest):
    model, fusion = get_pipeline()
    cutoff = ensure_utc(payload.as_of_date) if payload.as_of_date else datetime.now(timezone.utc)

    profile = CompanyProfile(
        company_id=payload.company_id,
        company_name=payload.company_name,
        snapshot_date=cutoff,
        total_funding_usd_m=payload.total_funding_usd_m,
        employee_count=payload.employee_count,
        growth_rate=payload.growth_rate,
        company_age_years=payload.company_age_years,
        industry_sector=payload.industry_sector,
    )

    domain_articles = [
        ArticleRecord(
            article_id=art.article_id,
            company_id=payload.company_id,
            source=art.source,
            title=art.title,
            content=art.content,
            published_at=ensure_utc(art.published_at),
        )
        for art in payload.articles
    ]

    try:
        fused_vector = fusion.fuse(profile, domain_articles, as_of_date=cutoff)
        features_dict = fused_vector.to_features_dict()
        df_in = pd.DataFrame([features_dict])

        probs = model.predict_proba(df_in)[0]
        pred_class = int(np.argmax(probs))
        class_labels = {0: "Distressed / High Risk", 1: "Stable / Moderate", 2: "High Growth"}

        return PredictionResponse(
            company_id=payload.company_id,
            predicted_class=pred_class,
            predicted_label=class_labels[pred_class],
            probabilities={
                "distressed": round(float(probs[0]), 4),
                "stable": round(float(probs[1]), 4),
                "high_growth": round(float(probs[2]), 4),
            },
            fused_features=features_dict,
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Inference pipeline execution failure: {str(exc)}",
        )


@app.post("/api/v1/explain", response_model=ExplainResponse)
def explain_startup_prediction(payload: AnalyzeRequest):
    model, fusion = get_pipeline()
    cutoff = ensure_utc(payload.as_of_date) if payload.as_of_date else datetime.now(timezone.utc)

    profile = CompanyProfile(
        company_id=payload.company_id,
        company_name=payload.company_name,
        snapshot_date=cutoff,
        total_funding_usd_m=payload.total_funding_usd_m,
        employee_count=payload.employee_count,
        growth_rate=payload.growth_rate,
        company_age_years=payload.company_age_years,
        industry_sector=payload.industry_sector,
    )

    fused_vector = fusion.fuse(profile, [], as_of_date=cutoff)
    df_in = pd.DataFrame([fused_vector.to_features_dict()])

    explainer = ModelExplainer(model=model, feature_names=list(df_in.columns))
    local_exp = explainer.explain_local_instance(df_in)

    return ExplainResponse(
        company_id=payload.company_id,
        predicted_class=local_exp["predicted_class"],
        top_positive_drivers=local_exp["top_positive_drivers"],
        top_negative_drags=local_exp["top_negative_drags"],
    )
"""
PIDE API: FastAPI REST Gateway
Provides a public interface for prompt injection detection.
"""

import logging
from fastapi import FastAPI, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from pipeline import load_pipeline, detect

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("PIDE.API")

app = FastAPI(
    title="PIDE — Prompt Injection Detection Engine",
    description="Multi-layer IS security engine: Regex · Semantic · Heuristic · Risk Scoring",
    version="1.0.0"
)

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global pipeline state
PIPELINE = None

class DetectRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=10000, description="User prompt to analyze")

class DetectResponse(BaseModel):
    decision: str
    risk_score: float
    l1_score: float
    l2_score: float
    l3_score: float
    trigger: str
    latency_ms: float
    l2_exemplars: List[str]
    l3_signals: Dict[str, float]

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool

@app.on_event("startup")
async def startup_event():
    global PIPELINE
    logger.info("Initializing PIDE Pipeline...")
    try:
        PIPELINE = load_pipeline()
        logger.info("Pipeline initialized successfully.")
    except Exception as e:
        logger.critical(f"Failed to initialize pipeline on startup: {e}")

@app.post("/detect", response_model=DetectResponse)
async def detect_prompt(request: DetectRequest):
    """Detects prompt injection in the provided text."""
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    try:
        result = detect(request.prompt, *PIPELINE)
        logger.info(f"API Detect: {result['decision']} (Risk: {result['risk_score']})")
        return result
    except Exception as e:
        logger.error(f"API Error: {e}")
        return {
            "decision": "BLOCK",
            "risk_score": 1.0,
            "l1_score": 1.0, "l2_score": 1.0, "l3_score": 1.0,
            "trigger": "API_EXCEPTION",
            "latency_ms": 0.0,
            "l2_exemplars": [],
            "l3_signals": {}
        }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Returns the health status of the service."""
    return {
        "status": "healthy",
        "version": "1.0.0",
        "model_loaded": PIPELINE is not None
    }

@app.get("/config")
async def get_config():
    """Returns the current scoring configuration (Safe fields only)."""
    if PIPELINE is None:
        raise HTTPException(status_code=503, detail="Pipeline not initialized")
    
    l4 = PIPELINE[3]
    return {
        "weights": l4.weights,
        "thresholds": l4.thresholds
    }

if __name__ == "__main__":
    import uvicorn
    # Run command: uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
    uvicorn.run(app, host="0.0.0.0", port=8000)

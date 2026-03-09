"""
API routes: upload CSV, run full pipeline, return results.
"""

import io

import pandas as pd
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.agents.orchestrator import AnalysisOrchestrator

router = APIRouter(prefix="/api", tags=["pipeline"])

MAX_SIZE_MB = 50


def _read_csv(file: UploadFile) -> pd.DataFrame:
    content = file.file.read()

    if len(content) > MAX_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max {MAX_SIZE_MB}MB.",
        )

    try:
        return pd.read_csv(io.BytesIO(content))
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid CSV: {e}",
        )


@router.post("/analyze")
async def analyze_csv(
    file: UploadFile = File(...),
    target_column: str | None = Form(default=None),
):
    """
    Upload a CSV and run the full AI Data Scientist pipeline:
    profiling, statistical insights, modeling, anomaly detection,
    cognitive flags, and executive summary.
    """

    if not file.filename or not file.filename.lower().endswith(".csv"):
        raise HTTPException(
            status_code=400,
            detail="Please upload a CSV file.",
        )

    df = _read_csv(file)

    if df.empty:
        raise HTTPException(
            status_code=400,
            detail="CSV is empty.",
        )

    orchestrator = AnalysisOrchestrator(
        df=df,
        target_column=target_column,
    )

    results = orchestrator.run()

    return JSONResponse(
        content={
            **results,
            "target_column": target_column,
        }
    )


@router.get("/health")
async def health():
    return {"status": "ok", "service": "ai-data-scientist"}
"""
Small FastAPI service for UH-DQIP

Endpoints:
- GET /anomalies
    * returns anomalies as JSON
    * supports optional query params: provider_id, start_date, end_date, severity
- GET /claim/{claim_id}/explanation
    * returns the anomaly record and explanation for given claim_id
    * calls the local explain_anomaly(record) from src.llm.explain if explanation missing
- Implementation details:
    * Load anomalies from data/gold/anomalies.csv (or data/gold/anomalies.parquet)
    * Cache the DataFrame in memory for performance (simple global cache)
    * Use Pydantic models for response shapes
    * Handle missing files gracefully with informative HTTP responses
    * Keep auth out for hackathon (can be added later) but read OPENAI_API_KEY from env for explanations
"""
from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Optional, Any
from pathlib import Path
import pandas as pd
import sys
import os

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

app = FastAPI(title="UH-DQIP API", version="0.1")

# Global cache for anomalies DataFrame
_anomalies_cache: Optional[pd.DataFrame] = None


# Pydantic models
class AnomalyRecord(BaseModel):
    claim_id: str
    provider_id: Optional[str] = None
    member_id: Optional[str] = None
    service_date: Optional[str] = None
    billed_amount: Optional[float] = None
    anomaly_reasons_str: Optional[str] = None
    num_flags: Optional[int] = None
    explanation: Optional[str] = None
    
    class Config:
        extra = "allow"


class AnomalyListResponse(BaseModel):
    total: int
    anomalies: List[dict]


class ExplanationResponse(BaseModel):
    claim_id: str
    anomaly_record: dict
    explanation: str


def load_anomalies() -> pd.DataFrame:
    """Load anomalies from gold layer, with caching."""
    global _anomalies_cache
    
    if _anomalies_cache is not None:
        return _anomalies_cache
    
    data_dir = Path("data/gold")
    parquet_file = data_dir / "anomalies.parquet"
    csv_file = data_dir / "anomalies.csv"
    
    if parquet_file.exists():
        _anomalies_cache = pd.read_parquet(parquet_file)
    elif csv_file.exists():
        _anomalies_cache = pd.read_csv(csv_file)
    else:
        raise FileNotFoundError("No anomalies data found. Run the pipeline first.")
    
    # Ensure claim_id is string
    if "claim_id" in _anomalies_cache.columns:
        _anomalies_cache["claim_id"] = _anomalies_cache["claim_id"].astype(str)
    
    return _anomalies_cache


@app.get("/", tags=["Health"])
def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "UH-DQIP API", "version": "0.1"}


@app.get("/anomalies", response_model=AnomalyListResponse, tags=["Anomalies"])
def get_anomalies(
    provider_id: Optional[str] = Query(None, description="Filter by provider ID"),
    start_date: Optional[str] = Query(None, description="Filter by start date (YYYY-MM-DD)"),
    end_date: Optional[str] = Query(None, description="Filter by end date (YYYY-MM-DD)"),
    severity: Optional[int] = Query(None, description="Minimum number of flags", ge=1),
    limit: int = Query(100, description="Max records to return", ge=1, le=1000)
):
    """
    Get list of anomalies with optional filters.
    """
    try:
        df = load_anomalies()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    filtered = df.copy()
    
    # Apply filters
    if provider_id:
        filtered = filtered[filtered["provider_id"].astype(str) == provider_id]
    
    if start_date and "service_date" in filtered.columns:
        filtered["service_date"] = pd.to_datetime(filtered["service_date"], errors="coerce")
        filtered = filtered[filtered["service_date"] >= pd.to_datetime(start_date)]
    
    if end_date and "service_date" in filtered.columns:
        filtered["service_date"] = pd.to_datetime(filtered["service_date"], errors="coerce")
        filtered = filtered[filtered["service_date"] <= pd.to_datetime(end_date)]
    
    if severity and "num_flags" in filtered.columns:
        filtered = filtered[filtered["num_flags"] >= severity]
    
    # Convert dates to string for JSON
    if "service_date" in filtered.columns:
        filtered["service_date"] = filtered["service_date"].astype(str)
    
    # Limit results
    filtered = filtered.head(limit)
    
    # Convert to dict records
    records = filtered.fillna("").to_dict(orient="records")
    
    return AnomalyListResponse(total=len(records), anomalies=records)


@app.get("/claim/{claim_id}/explanation", response_model=ExplanationResponse, tags=["Explanations"])
def get_claim_explanation(claim_id: str):
    """
    Get anomaly record and explanation for a specific claim.
    If explanation is missing, generates it using LLM.
    """
    try:
        df = load_anomalies()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    # Find the claim
    claim_record = df[df["claim_id"] == claim_id]
    
    if claim_record.empty:
        raise HTTPException(status_code=404, detail=f"Claim {claim_id} not found in anomalies")
    
    record = claim_record.iloc[0].to_dict()
    
    # Check if explanation exists
    explanation = record.get("explanation", "")
    
    if not explanation or pd.isna(explanation) or explanation == "":
        # Generate explanation using LLM
        try:
            from src.llm.explain import explain_anomaly
            explanation = explain_anomaly(record)
        except Exception as e:
            explanation = f"Could not generate explanation: {str(e)}"
    
    # Clean up NaN values for JSON
    clean_record = {k: (v if pd.notna(v) else None) for k, v in record.items()}
    
    return ExplanationResponse(
        claim_id=claim_id,
        anomaly_record=clean_record,
        explanation=explanation
    )


@app.get("/stats", tags=["Statistics"])
def get_stats():
    """Get summary statistics about anomalies."""
    try:
        df = load_anomalies()
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    
    stats = {
        "total_anomalies": len(df),
        "unique_providers": df["provider_id"].nunique() if "provider_id" in df.columns else 0,
        "unique_members": df["member_id"].nunique() if "member_id" in df.columns else 0,
    }
    
    if "num_flags" in df.columns:
        stats["avg_flags_per_anomaly"] = round(df["num_flags"].mean(), 2)
        stats["max_flags"] = int(df["num_flags"].max())
    
    if "anomaly_reasons_str" in df.columns:
        all_reasons = []
        for reasons_str in df["anomaly_reasons_str"].dropna():
            if reasons_str:
                all_reasons.extend(reasons_str.split(","))
        reason_counts = pd.Series(all_reasons).value_counts().to_dict()
        stats["anomaly_types"] = reason_counts
    
    return stats


@app.post("/refresh", tags=["Admin"])
def refresh_cache():
    """Clear the cache and reload data."""
    global _anomalies_cache
    _anomalies_cache = None
    
    try:
        load_anomalies()
        return {"status": "ok", "message": "Cache refreshed successfully"}
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))


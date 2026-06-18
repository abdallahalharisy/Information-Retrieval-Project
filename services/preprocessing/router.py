"""Preprocessing Service API."""

from fastapi import APIRouter, HTTPException

from shared.schemas import (
    HealthResponse,
    PreprocessBatchRequest,
    PreprocessBatchResponse,
    PreprocessDocumentRequest,
    PreprocessDocumentResponse,
)
from services.preprocessing.service import preprocess_batch, preprocess_document

router = APIRouter(prefix="/preprocessing", tags=["Preprocessing"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(service="preprocessing")


@router.post("/document", response_model=PreprocessDocumentResponse)
def preprocess_one(request: PreprocessDocumentRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Empty text")
    tokens = preprocess_document(request.text)
    return PreprocessDocumentResponse(tokens=tokens)


@router.post("/batch", response_model=PreprocessBatchResponse)
def preprocess_many(request: PreprocessBatchRequest):
    processed = preprocess_batch(request.documents)
    return PreprocessBatchResponse(processed=processed, count=len(processed))

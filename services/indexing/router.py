"""Indexing Service API."""

from fastapi import APIRouter, HTTPException

from shared.schemas import HealthResponse, IndexBuildRequest, IndexStatusResponse
from services.indexing.service import build_index, get_all_status, get_index_status

router = APIRouter(prefix="/indexing", tags=["Indexing"])


@router.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(service="indexing")


@router.post("/build", response_model=IndexStatusResponse)
def build(request: IndexBuildRequest):
    try:
        result = build_index(
            request.dataset,
            fit_word2vec=request.fit_word2vec,
            fit_bert=request.fit_bert,
        )
        return IndexStatusResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@router.get("/status/{dataset}", response_model=IndexStatusResponse)
def status(dataset: str):
    try:
        return IndexStatusResponse(**get_index_status(dataset))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/status")
def status_all():
    return get_all_status()

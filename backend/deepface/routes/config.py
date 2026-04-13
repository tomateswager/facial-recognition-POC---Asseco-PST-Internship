"""
Rotas de configuração do DeepFace — leitura e escrita de thresholds em runtime.
"""
from fastapi import APIRouter
from pydantic import BaseModel

from backend.config import (
    VERIFY_THRESHOLD,
    IDENTIFY_THRESHOLD,
    DOCUMENT_THRESHOLD,
    QUALITY_MAX_BLUR,
    QUALITY_MIN_FACE_SIZE,
    get_df_runtime_config,
    save_df_runtime_config,
)

router = APIRouter(prefix="/api/deepface/config", tags=["Configuração DeepFace"])

_DEFAULTS = {
    "df_verify_threshold":      VERIFY_THRESHOLD,
    "df_identify_threshold":    IDENTIFY_THRESHOLD,
    "df_document_threshold":    DOCUMENT_THRESHOLD,
    "df_quality_min_blur":      QUALITY_MAX_BLUR,
    "df_quality_min_face_size": QUALITY_MIN_FACE_SIZE,
}


@router.get("")
async def get_config():
    cfg = get_df_runtime_config()
    return {
        "df_verify_threshold":      cfg["df_verify_threshold"],
        "df_identify_threshold":    cfg["df_identify_threshold"],
        "df_document_threshold":    cfg["df_document_threshold"],
        "df_quality_min_blur":      cfg["df_quality_min_blur"],
        "df_quality_min_face_size": int(cfg["df_quality_min_face_size"]),
        "defaults": _DEFAULTS,
    }


class DFConfigUpdate(BaseModel):
    df_verify_threshold:      float | None = None
    df_identify_threshold:    float | None = None
    df_document_threshold:    float | None = None
    df_quality_min_blur:      float | None = None
    df_quality_min_face_size: int   | None = None


@router.post("")
async def update_config(body: DFConfigUpdate):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        return {"success": False, "detail": "Nenhum valor fornecido."}
    save_df_runtime_config(updates)
    return {"success": True, "saved": updates}

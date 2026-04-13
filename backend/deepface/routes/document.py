"""
Rotas de Extração de Fotografia de Documento de Identificação
Fluxo tipo banco: upload documento → extração da face → câmara → verificação 1:1.
Objetivo 5 do POC.
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.config import MAX_IMAGE_BYTES
from backend.deepface.services.document_service import (
    validate_document_image,
    extract_face_from_document,
    verify_document_vs_live,
    cleanup_session,
    cleanup_old_sessions,
)

router = APIRouter(prefix="/api/document", tags=["Documento de Identificação"])


@router.post("/validate")
async def validate_document(
    document: UploadFile = File(..., description="Imagem do documento a validar"),
):
    """
    Validação rápida do documento antes da extração.
    Verifica resolução, orientação e presença de face.
    """
    doc_bytes = await document.read()
    if not doc_bytes:
        raise HTTPException(status_code=400, detail="A imagem do documento é obrigatória.")
    if len(doc_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem demasiado grande (máximo 10 MB).")
    return validate_document_image(doc_bytes)


@router.post("/extract")
async def extract_from_document(
    document: UploadFile = File(..., description="Imagem do documento de identificação (CC, Passaporte, Carta)"),
):
    """
    Passo 1: Extrai a fotografia do titular a partir do documento.
    Retorna a face recortada, métricas de qualidade e um session_id para o passo 2.
    """
    doc_bytes = await document.read()
    if not doc_bytes:
        raise HTTPException(status_code=400, detail="A imagem do documento é obrigatória.")
    if len(doc_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem demasiado grande (máximo 10 MB).")

    result = extract_face_from_document(doc_bytes)
    return result


@router.post("/verify-live")
async def verify_live_face(
    image: UploadFile = File(..., description="Imagem capturada em tempo real pela câmara"),
    session_id: str = Form(..., description="ID da sessão (do passo de extração)"),
):
    """
    Passo 2: Verifica se a face da câmara corresponde à face do documento.
    Anti-spoofing ativado (câmara em tempo real).
    """
    live_bytes = await image.read()
    if not live_bytes:
        raise HTTPException(status_code=400, detail="A imagem da câmara é obrigatória.")
    if len(live_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem demasiado grande (máximo 10 MB).")

    if not session_id or not session_id.strip():
        raise HTTPException(status_code=400, detail="Session ID é obrigatório.")

    result = verify_document_vs_live(session_id.strip(), live_bytes)
    return result


@router.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Limpa dados temporários de uma sessão de verificação de documento."""
    return cleanup_session(session_id)


@router.post("/session/{session_id}/cleanup")
async def cleanup_session_beacon(session_id: str):
    """Endpoint POST para limpeza via sendBeacon (chamado no beforeunload do browser)."""
    return cleanup_session(session_id)


@router.post("/sessions/cleanup-old")
async def cleanup_old_sessions_route():
    """Remove sessões de documento com mais de 30 minutos."""
    removed = cleanup_old_sessions(max_age_minutes=30)
    return {"success": True, "removed": removed}

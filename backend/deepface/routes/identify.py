"""
Rotas de Identificação 1:N
Pesquisa um rosto na base de dados de rostos registados.
"""
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.deepface.services.deepface_service import identify_1toN
from backend.config import MAX_IMAGE_BYTES

router = APIRouter(prefix="/api/identify", tags=["Identificação 1:N"])


@router.post("/")
async def identify_face(
    image: UploadFile = File(..., description="Imagem facial a identificar"),
    mode: str = Form("file", description="Modo: 'file' ou 'camera'"),
):
    """
    Identificação 1:N - Pesquisa um rosto na base de dados.

    mode=file → Anti-spoofing desativado
    mode=camera → Anti-spoofing ativado
    """
    image_bytes = await image.read()

    if not image_bytes:
        raise HTTPException(status_code=400, detail="A imagem é obrigatória.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem demasiado grande (máximo 10 MB).")

    use_anti_spoofing = (mode == "camera")
    t0 = time.perf_counter()
    result = identify_1toN(image_bytes, use_anti_spoofing=use_anti_spoofing)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return result

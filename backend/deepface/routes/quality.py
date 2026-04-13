"""
Rotas de Avaliação de Qualidade de Imagem
Analisa atributos de qualidade: desfocagem, brilho, tamanho, orientação.
Objetivo 4 do POC.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.deepface.services.quality_service import full_quality_assessment
from backend.config import MAX_IMAGE_BYTES

router = APIRouter(prefix="/api/quality", tags=["Qualidade de Imagem"])


@router.post("/assess")
async def assess_quality(
    image: UploadFile = File(..., description="Imagem para avaliação de qualidade"),
):
    """
    Avaliação completa de qualidade de imagem facial.
    Verifica: resolução, desfocagem, brilho, contraste, tamanho da face, confiança.
    """
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="A imagem é obrigatória.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem demasiado grande (máximo 10 MB).")

    result = full_quality_assessment(image_bytes)
    return result

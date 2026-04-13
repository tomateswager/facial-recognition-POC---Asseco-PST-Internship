"""
Rotas de Avaliação Comparativa de Modelos
Testa e compara modelos de reconhecimento facial.
Objetivo 3 do POC.
"""
from fastapi import APIRouter, UploadFile, File, HTTPException
from backend.deepface.services.model_comparison_service import evaluate_models_verify, get_model_info

router = APIRouter(prefix="/api/models", tags=["Avaliação de Modelos"])


@router.post("/compare")
async def compare_models(
    image1: UploadFile = File(..., description="Primeira imagem facial"),
    image2: UploadFile = File(..., description="Segunda imagem facial"),
):
    """
    Avalia todos os modelos configurados numa verificação 1:1.
    Retorna matriz comparativa com precisão, tempo e recursos.
    """
    bytes1 = await image1.read()
    bytes2 = await image2.read()

    if not bytes1 or not bytes2:
        raise HTTPException(status_code=400, detail="Ambas as imagens são obrigatórias.")

    result = evaluate_models_verify(bytes1, bytes2)
    return result


@router.get("/info")
async def models_info():
    """Retorna informação teórica sobre os modelos disponíveis."""
    return {
        "success": True,
        "models": get_model_info(),
    }

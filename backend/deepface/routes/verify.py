"""
Rotas de Verificacao 1:1
Compara dois rostos para determinar se pertencem a mesma pessoa.
"""
import time
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.deepface.services.deepface_service import verify_1to1
from backend.config import MAX_IMAGE_BYTES

router = APIRouter(prefix='/api/verify', tags=['Verificacao 1:1'])


@router.post('/')
async def verify_faces(
    image1: UploadFile = File(..., description='Primeira imagem facial'),
    image2: UploadFile = File(..., description='Segunda imagem facial'),
    mode: str = Form('file', description="Modo: 'file' ou 'camera'"),
):
    """
    Verificacao 1:1.

    mode=file   -> compara duas imagens sem anti-spoofing.
    mode=camera -> ativa anti-spoofing na imagem ao vivo.
    """
    bytes1 = await image1.read()
    bytes2 = await image2.read()

    if not bytes1 or not bytes2:
        raise HTTPException(status_code=400, detail='Ambas as imagens sao obrigatorias.')
    if len(bytes1) > MAX_IMAGE_BYTES or len(bytes2) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail='Imagem demasiado grande (máximo 10 MB).')

    # Em modo camera, ativa anti-spoofing do DeepFace na imagem ao vivo.
    # (No service, a validacao e aplicada ao frame ao vivo antes da comparacao 1:1.)
    use_anti_spoofing = (mode == 'camera')
    t0 = time.perf_counter()
    result = verify_1to1(bytes1, bytes2, use_anti_spoofing=use_anti_spoofing)
    result["latency_ms"] = round((time.perf_counter() - t0) * 1000)
    return result

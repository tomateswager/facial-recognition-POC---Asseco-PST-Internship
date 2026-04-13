"""
Rota de Análise de Atributos Faciais
POST /api/attributes — analisa idade, género e emoções de uma imagem
"""
from fastapi import APIRouter, UploadFile, File
from backend.deepface.services.deepface_service import analyze_face_attributes

router = APIRouter(prefix="/api/attributes", tags=["Atributos Faciais"])


@router.post("/")
async def face_attributes(image: UploadFile = File(...)):
    image_bytes = await image.read()
    if not image_bytes:
        return {"success": False, "message": "Imagem obrigatória"}
    return analyze_face_attributes(image_bytes)

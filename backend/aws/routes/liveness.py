"""
Rotas AWS - Face Liveness
POST /api/aws/liveness/create  → cria sessão de liveness
GET  /api/aws/liveness/result/{session_id} → resultado da sessão
"""
import time
import requests
from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/aws", tags=["aws"])

_LIVENESS_PORT = 5001


@router.post("/liveness/create")
async def create_liveness_session():
    from backend.aws.services.rekognition_service import RekognitionClient
    client = RekognitionClient()
    result = client.create_face_liveness_session()
    return {"success": True, "session_id": result["session_id"]}


@router.get("/liveness/result/{session_id}")
async def get_liveness_result(session_id: str):
    try:
        resp = requests.get(f"http://localhost:{_LIVENESS_PORT}/api/result/{session_id}", timeout=10)
        return resp.json()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Servidor de liveness não disponível: {e}")

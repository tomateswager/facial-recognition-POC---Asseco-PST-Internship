"""
Rotas AWS Rekognition - Identificação 1:N
POST /api/aws/identify
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.aws.services.rekognition_service import RekognitionClient
from backend.config import AWS_COLLECTION_ID, AWS_IDENTIFY_THRESHOLD

router = APIRouter(prefix="/api/aws", tags=["aws"])


@router.post("/identify")
async def aws_identify(
    file: UploadFile = File(...),
    max_faces: int = Form(default=5),
):
    client = RekognitionClient()
    image_bytes = await file.read()
    result = client.search_faces_by_image(
        collection_id=AWS_COLLECTION_ID,
        image_bytes=image_bytes,
        max_faces=max_faces,
        threshold=AWS_IDENTIFY_THRESHOLD,
    )
    from backend.aws.services.metrics_service import get_collector
    top_sim = result["matches"][0]["similarity"] if result["matches"] else 0
    get_collector().record("SearchFacesByImage", result["latency_ms"],
                           success=result["match_count"] > 0, top_similarity=top_sim)
    return {
        "success": True,
        "match_count": result["match_count"],
        "matches": result["matches"],
        "searched_face_confidence": result["searched_face_confidence"],
        "threshold": result["threshold_used"],
        "latency_ms": result["latency_ms"],
    }

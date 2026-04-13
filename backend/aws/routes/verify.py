"""
Rotas AWS Rekognition - Verificação 1:1
POST /api/aws/verify  — compara dois uploads
POST /api/aws/verify/by-name — compara upload contra face registada na coleção
POST /api/aws/attributes — analisa atributos faciais (idade, género, emoção)
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.aws.services.rekognition_service import RekognitionClient
from backend.config import AWS_COLLECTION_ID, AWS_VERIFY_THRESHOLD

router = APIRouter(prefix="/api/aws", tags=["aws"])


@router.post("/verify")
async def aws_verify(
    file1: UploadFile = File(...),
    file2: UploadFile = File(...),
):
    client = RekognitionClient()
    b1 = await file1.read()
    b2 = await file2.read()
    result = client.compare_faces(source_bytes=b1, target_bytes=b2, similarity_threshold=AWS_VERIFY_THRESHOLD)
    from backend.aws.services.metrics_service import get_collector
    get_collector().record("CompareFaces", result["latency_ms"], success=result["is_match"],
                           similarity=result["best_similarity"])
    return {
        "success": True,
        "is_match": result["is_match"],
        "similarity": result["best_similarity"],
        "threshold": result["threshold_used"],
        "latency_ms": result["latency_ms"],
    }


@router.post("/verify/by-name")
async def aws_verify_by_name(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    client = RekognitionClient()
    image_bytes = await file.read()
    result = client.search_faces_by_image(
        collection_id=AWS_COLLECTION_ID,
        image_bytes=image_bytes,
        max_faces=20,
        threshold=0.0,
    )
    import unicodedata
    def _aws_name(n):
        normalized = unicodedata.normalize("NFKD", n.strip()).encode("ascii", "ignore").decode()
        return "".join(c if c.isalnum() or c in "_.-:" else "_" for c in normalized).strip("_")
    target = _aws_name(name)
    match = next((m for m in result.get("matches", []) if m["external_id"] == target), None)
    is_match = match is not None and match["similarity"] >= AWS_VERIFY_THRESHOLD
    return {
        "success": True,
        "is_match": is_match,
        "person": name,
        "similarity": match["similarity"] if match else 0.0,
        "threshold": AWS_VERIFY_THRESHOLD,
        "latency_ms": result["latency_ms"],
    }


@router.post("/attributes")
async def aws_face_attributes(image: UploadFile = File(...)):
    """Analisa atributos faciais via AWS Rekognition DetectFaces."""
    try:
        client = RekognitionClient()
    except ValueError as e:
        return {"success": False, "message": str(e)}
    image_bytes = await image.read()
    if not image_bytes:
        return {"success": False, "message": "Imagem obrigatória"}
    try:
        result = client.detect_faces(image_bytes)
    except Exception as e:
        return {"success": False, "message": str(e)}

    if not result.get("faces"):
        return {"success": False, "message": "Nenhuma face detetada", "latency_ms": result.get("latency_ms", 0)}

    face = result["faces"][0]
    emotions_raw = face.get("emotions", [])
    dominant = emotions_raw[0]["type"].lower() if emotions_raw else "neutral"

    _EMOJI = {"happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
              "surprised": "😲", "disgusted": "🤢", "calm": "😐", "confused": "🤔"}

    return {
        "success": True,
        "age_range": face.get("age_range", {}),
        "gender": face.get("gender"),
        "gender_confidence": round(face.get("gender_confidence", 0), 1),
        "dominant_emotion": dominant,
        "emotion_emoji": _EMOJI.get(dominant, "😐"),
        "emotions": [{"name": e["type"].lower(), "confidence": round(e["confidence"], 1)} for e in emotions_raw[:5]],
        "smile": face.get("smile"),
        "eyes_open": face.get("eyes_open"),
        "latency_ms": result.get("latency_ms", 0),
    }

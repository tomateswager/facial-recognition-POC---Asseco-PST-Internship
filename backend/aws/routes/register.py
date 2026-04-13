"""
Rotas AWS Rekognition - Registo de Face
POST /api/aws/register
DELETE /api/aws/register/{person_name}
GET /api/aws/register/list
"""
import unicodedata
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.aws.services.rekognition_service import RekognitionClient
from backend.config import AWS_COLLECTION_ID

router = APIRouter(prefix="/api/aws", tags=["aws"])


def _aws_name(name: str) -> str:
    normalized = unicodedata.normalize("NFKD", name.strip()).encode("ascii", "ignore").decode()
    return "".join(c if c.isalnum() or c in "_.-:" else "_" for c in normalized).strip("_")


@router.post("/register")
async def aws_register(
    name: str = Form(...),
    file: UploadFile = File(...),
):
    client = RekognitionClient()
    client.create_collection(AWS_COLLECTION_ID)
    image_bytes = await file.read()
    person_name = _aws_name(name)
    if not person_name:
        raise HTTPException(status_code=400, detail="Nome inválido")
    detection = client.detect_faces(image_bytes)
    if detection["face_count"] == 0:
        raise HTTPException(status_code=400, detail="Nenhuma face detetada na imagem")
    result = client.index_face(collection_id=AWS_COLLECTION_ID, image_bytes=image_bytes, external_id=person_name)
    if not result["success"]:
        raise HTTPException(status_code=400, detail="Face não indexada na coleção AWS")
    from backend.aws.services.metrics_service import get_collector
    get_collector().record("IndexFaces", result["latency_ms"], success=result["success"])
    return {
        "success": True,
        "person": person_name,
        "face_id": result["indexed_faces"][0]["face_id"] if result["indexed_faces"] else "",
        "confidence": result["indexed_faces"][0]["confidence"] if result["indexed_faces"] else 0,
        "latency_ms": result["latency_ms"],
    }


@router.delete("/register/{person_name}")
async def aws_delete_face(person_name: str):
    client = RekognitionClient()
    faces = client.list_faces_in_collection(AWS_COLLECTION_ID)
    target = _aws_name(person_name)
    face_ids = [f["FaceId"] for f in faces if f.get("ExternalImageId") == target]
    if not face_ids:
        raise HTTPException(status_code=404, detail=f"'{person_name}' não encontrado na coleção")
    result = client.delete_face(AWS_COLLECTION_ID, *face_ids)
    return {"success": True, "deleted": result["deleted"], "latency_ms": result["latency_ms"]}


@router.get("/register/list")
async def aws_list_faces():
    client = RekognitionClient()
    faces = client.list_faces_in_collection(AWS_COLLECTION_ID)
    return {
        "success": True,
        "faces": [{"face_id": f["FaceId"], "name": f.get("ExternalImageId", ""), "confidence": f.get("Confidence", 0)} for f in faces],
        "total": len(faces),
    }

"""
Rota Multi-Pessoa — Detecção e selecção de face em grupo
POST /api/multi/detect   — detecta todas as faces numa imagem (AWS ou DeepFace)
"""
import io
import os
import time

from fastapi import APIRouter, UploadFile, File, Query
from PIL import Image

from backend.config import TEMP_DIR

router = APIRouter(prefix="/api/multi", tags=["Multi-Pessoa"])


def _sort_faces_left_to_right(faces: list[dict]) -> list[dict]:
    """
    Ordena faces da esquerda para a direita.
    Em caso de empate, usa top para manter ordem estavel.
    """
    sorted_faces = sorted(
        faces,
        key=lambda face: (
            face.get("bbox", {}).get("left", 0),
            face.get("bbox", {}).get("top", 0),
        ),
    )
    for i, face in enumerate(sorted_faces):
        face["id"] = i
    return sorted_faces


@router.post("/detect")
async def detect_multi(
    file: UploadFile = File(...),
    engine: str = Query("aws", description="'aws' ou 'deepface'"),
):
    image_bytes = await file.read()

    if engine == "aws":
        try:
            from backend.aws.services.rekognition_service import RekognitionClient
            client = RekognitionClient()
            result = client.detect_faces(image_bytes)
        except Exception as e:
            return {"success": False, "detail": str(e), "faces": [], "face_count": 0}
        faces = []
        for i, f in enumerate(result["faces"]):
            bb = f["bounding_box"]  # {Left, Top, Width, Height} — frações 0-1
            faces.append({
                "id": i,
                "bbox": {
                    "left":   bb.get("Left", 0),
                    "top":    bb.get("Top", 0),
                    "width":  bb.get("Width", 0),
                    "height": bb.get("Height", 0),
                },
                "confidence": round(f["confidence"], 1),
            })
        faces = _sort_faces_left_to_right(faces)
        return {
            "success": True,
            "engine": "aws",
            "face_count": len(faces),
            "faces": faces,
            "latency_ms": result["latency_ms"],
        }

    else:  # deepface
        from deepface import DeepFace
        from backend.deepface.services.deepface_service import _save_temp_image

        t0 = time.time()
        temp_path = _save_temp_image(image_bytes)
        # Usar dimensões da imagem GUARDADA (pode ter sido redimensionada)
        saved_img = Image.open(temp_path)
        img_w, img_h = saved_img.size
        saved_img.close()
        try:
            faces_raw = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend="yunet",
                enforce_detection=False,
                align=False,
            )
        except Exception:
            faces_raw = []
        finally:
            try:
                os.remove(temp_path)
            except Exception:
                pass

        latency = (time.time() - t0) * 1000
        faces = []
        for i, f in enumerate(faces_raw):
            area = f.get("facial_area", {})
            x = area.get("x", 0)
            y = area.get("y", 0)
            w = area.get("w", 0)
            h = area.get("h", 0)
            conf = f.get("confidence") or 0
            if w > 0 and h > 0:
                faces.append({
                    "id": i,
                    "bbox": {
                        "left":   x / img_w,
                        "top":    y / img_h,
                        "width":  w / img_w,
                        "height": h / img_h,
                    },
                    "confidence": round(float(conf) * 100, 1),
                })

        faces = _sort_faces_left_to_right(faces)
        return {
            "success": True,
            "engine": "deepface",
            "face_count": len(faces),
            "faces": faces,
            "latency_ms": round(latency, 2),
        }

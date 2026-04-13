"""
Rota de Câmara ao Vivo
POST /api/live/analyze — analisa um frame da câmara:
  1. Verifica nitidez (Laplacian)
  2. Detecta faces com bounding boxes
  3. Anti-spoofing por face (se torch disponível)
  4. Identificação 1:N por face (só se real)
"""
import io
import os

import cv2
import numpy as np
from PIL import Image
from deepface import DeepFace
from fastapi import APIRouter, UploadFile, File

from backend.deepface.services.deepface_service import (
    identify_1toN,
    _save_temp_image,
    _get_anti_spoofing_runtime_status,
)

router = APIRouter(prefix="/api/live", tags=["Câmara Live"])

_MIN_SHARPNESS    = 30.0   # abaixo disto → frame desfocado, skip
_MIN_FACE_PX      = 40     # face menor que isto → ignorar
_FACE_MARGIN      = 0.18   # margem ao redor do crop da face


def _sharpness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _crop_face(img_bgr: np.ndarray, x: int, y: int, fw: int, fh: int) -> np.ndarray:
    h, w = img_bgr.shape[:2]
    mx = int(fw * _FACE_MARGIN)
    my = int(fh * _FACE_MARGIN)
    cx = max(0, x - mx)
    cy = max(0, y - my)
    cw = min(w - cx, fw + 2 * mx)
    ch = min(h - cy, fh + 2 * my)
    return img_bgr[cy:cy + ch, cx:cx + cw]


def _bgr_to_bytes(img_bgr: np.ndarray, quality: int = 90) -> bytes:
    rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


@router.post("/analyze")
async def live_analyze(file: UploadFile = File(...)):
    image_bytes = await file.read()

    # Decode
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return {"error": "Imagem inválida", "faces": []}

    h, w = img.shape[:2]

    # 1. Nitidez global
    sharp = _sharpness(img)
    if sharp < _MIN_SHARPNESS:
        return {"blurry": True, "sharpness": round(sharp, 1), "faces": []}

    # 2. Detectar faces
    temp_path = _save_temp_image(image_bytes)
    try:
        try:
            faces_raw = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend="yunet",
                enforce_detection=False,
                align=True,
                anti_spoofing=False,
            )
        except Exception:
            faces_raw = []
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    if not faces_raw:
        return {"blurry": False, "sharpness": round(sharp, 1), "faces": []}

    anti_spoof_available = _get_anti_spoofing_runtime_status()
    results = []

    for face_data in faces_raw:
        area = face_data.get("facial_area", {})
        x  = area.get("x", 0)
        y  = area.get("y", 0)
        fw = area.get("w", 0)
        fh = area.get("h", 0)
        conf = float(face_data.get("confidence") or 0)

        if fw < _MIN_FACE_PX or fh < _MIN_FACE_PX:
            continue

        bbox = {
            "left":   x / w,
            "top":    y / h,
            "width":  fw / w,
            "height": fh / h,
        }

        # 3. Crop da face
        face_crop = _crop_face(img, x, y, fw, fh)
        face_sharp = _sharpness(face_crop)
        face_bytes = _bgr_to_bytes(face_crop)

        # 4. Anti-spoofing
        is_real = None
        if anti_spoof_available:
            face_temp = _save_temp_image(face_bytes)
            try:
                spoof_faces = DeepFace.extract_faces(
                    img_path=face_temp,
                    detector_backend="opencv",
                    enforce_detection=False,
                    align=False,
                    anti_spoofing=True,
                )
                if spoof_faces:
                    is_real = bool(spoof_faces[0].get("is_real", True))
            except Exception:
                is_real = None
            finally:
                try:
                    os.remove(face_temp)
                except Exception:
                    pass

        # 5. Identificação (só se face real ou anti-spoofing indisponível)
        identity  = None
        similarity = 0.0
        status    = "analyzing"

        if is_real is False:
            status = "spoof"
        else:
            if face_sharp < _MIN_SHARPNESS:
                status = "blurry_face"
            else:
                try:
                    id_result = identify_1toN(face_bytes, use_anti_spoofing=False)
                    if id_result.get("success") and id_result.get("results"):
                        best      = id_result["results"][0]
                        similarity = round((1 - best.get("distance", 1)) * 100, 1)
                        identity  = best.get("name")
                        status    = "identified"
                    else:
                        status = "not_found"
                except Exception:
                    status = "not_found"

        results.append({
            "bbox":        bbox,
            "is_real":     is_real,
            "status":      status,       # identified | not_found | spoof | blurry_face | analyzing
            "identity":    identity,
            "similarity":  similarity,
            "confidence":  round(conf * 100, 1),
            "face_sharp":  round(face_sharp, 1),
        })

    return {
        "blurry":    False,
        "sharpness": round(sharp, 1),
        "faces":     results,
    }

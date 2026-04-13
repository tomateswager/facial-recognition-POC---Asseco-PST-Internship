"""
Rotas AWS - Configuração e Estatísticas
GET  /api/aws/config         → configuração atual
POST /api/aws/config         → atualiza thresholds
GET  /api/aws/stats          → estatísticas da coleção + ligação
"""
import json
import os
import time
import requests as _req
from fastapi import APIRouter, HTTPException, UploadFile, File
from pydantic import BaseModel
from backend.config import AWS_COLLECTION_ID

router = APIRouter(prefix="/api/aws", tags=["aws"])

_CONFIG_FILE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))), "config.json")


def _read_config() -> dict:
    defaults = {
        "verify_threshold": float(os.getenv("AWS_VERIFY_THRESHOLD", "95.0")),
        "identify_threshold": float(os.getenv("AWS_IDENTIFY_THRESHOLD", "80.0")),
        "liveness_threshold": 75.0,
    }
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            stored = json.load(f)
        defaults.update({k: stored[k] for k in defaults if k in stored})
    except Exception:
        pass
    return defaults


def _write_config(data: dict):
    try:
        existing = {}
        try:
            with open(_CONFIG_FILE, encoding="utf-8") as f:
                existing = json.load(f)
        except Exception:
            pass
        existing.update(data)
        with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
            json.dump(existing, f, indent=2)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao guardar configuração: {e}")


class ConfigUpdate(BaseModel):
    verify_threshold: float | None = None
    identify_threshold: float | None = None
    liveness_threshold: float | None = None


@router.get("/config")
async def get_aws_config():
    cfg = _read_config()
    liveness_ok = False
    try:
        r = _req.get("http://localhost:5001/health", timeout=1)
        liveness_ok = r.status_code == 200
    except Exception:
        pass
    return {
        "success": True,
        "collection_id": AWS_COLLECTION_ID,
        "region": os.getenv("AWS_REGION", "eu-west-1"),
        "liveness_server_ok": liveness_ok,
        **cfg,
    }


@router.post("/config")
async def update_aws_config(body: ConfigUpdate):
    updates = {k: v for k, v in body.model_dump().items() if v is not None}
    if not updates:
        raise HTTPException(status_code=400, detail="Nenhum valor para atualizar")
    # Validate ranges
    for key in ["verify_threshold", "identify_threshold", "liveness_threshold"]:
        if key in updates and not (0.0 <= updates[key] <= 100.0):
            raise HTTPException(status_code=400, detail=f"{key} deve estar entre 0 e 100")
    _write_config(updates)
    return {"success": True, "updated": updates}


@router.post("/detect")
async def aws_detect(file: UploadFile = File(...)):
    from backend.aws.services.rekognition_service import RekognitionClient
    client = RekognitionClient()
    image_bytes = await file.read()
    result = client.detect_faces(image_bytes)
    return {"success": True, **result}


@router.get("/metrics")
async def get_aws_metrics():
    from backend.aws.services.metrics_service import get_collector
    col = get_collector()
    summary = col.get_summary()
    recent = col.get_recent(50)

    # Separate verify vs identify stats
    verify_ops = [m for m in recent if m["operation"] == "CompareFaces"]
    identify_ops = [m for m in recent if m["operation"] == "SearchFacesByImage"]

    verify_matches = [m for m in verify_ops if m.get("success")]
    verify_no_match = [m for m in verify_ops if not m.get("success")]
    identify_hits = [m for m in identify_ops if m.get("success")]
    identify_miss = [m for m in identify_ops if not m.get("success")]

    avg_sim_verify = (sum(m.get("similarity", 0) for m in verify_ops) / len(verify_ops)) if verify_ops else 0
    avg_sim_identify = (sum(m.get("top_similarity", 0) for m in identify_hits) / len(identify_hits)) if identify_hits else 0

    # Confidence scores from all ops
    confidences = [m.get("similarity") or m.get("top_similarity") for m in recent
                   if m.get("similarity") is not None or m.get("top_similarity") is not None]

    return {
        "success": True,
        "summary": summary,
        "recent": recent,
        "verification": {
            "total": len(verify_ops),
            "matches": len(verify_matches),
            "no_matches": len(verify_no_match),
            "success_rate": round(len(verify_matches) / len(verify_ops) * 100, 1) if verify_ops else 0,
            "avg_similarity": round(avg_sim_verify, 1),
        },
        "identification": {
            "total": len(identify_ops),
            "hits": len(identify_hits),
            "misses": len(identify_miss),
            "hit_rate": round(len(identify_hits) / len(identify_ops) * 100, 1) if identify_ops else 0,
            "avg_top_similarity": round(avg_sim_identify, 1),
        },
        "confidences": confidences,
    }


@router.post("/metrics/clear")
async def clear_aws_metrics():
    from backend.aws.services.metrics_service import get_collector
    get_collector().reset()
    return {"success": True}


@router.get("/stats")
async def get_aws_stats():
    from backend.aws.services.rekognition_service import RekognitionClient
    try:
        client = RekognitionClient()
    except ValueError as e:
        raise HTTPException(status_code=503, detail=str(e))

    # Connection test
    conn = client.test_connection()

    # Collection faces
    faces = []
    face_error = None
    try:
        raw = client.list_faces_in_collection(AWS_COLLECTION_ID)
        seen = set()
        for f in raw:
            name = f.get("ExternalImageId", "?")
            if name not in seen:
                seen.add(name)
                faces.append({
                    "name": name,
                    "face_id": f.get("FaceId", ""),
                    "confidence": round(f.get("Confidence", 0), 1),
                })
    except Exception as e:
        face_error = str(e)

    # Liveness server health
    liveness_ok = False
    try:
        r = _req.get("http://localhost:5001/api/health", timeout=2)
        liveness_ok = r.status_code == 200
    except Exception:
        pass

    cfg = _read_config()

    return {
        "success": True,
        "aws_connected": conn.get("success", False),
        "aws_latency_ms": conn.get("latency_ms"),
        "aws_region": conn.get("region", os.getenv("AWS_REGION", "eu-west-1")),
        "collection_id": AWS_COLLECTION_ID,
        "total_unique_people": len(faces),
        "total_face_entries": len(faces),
        "people": faces,
        "face_error": face_error,
        "liveness_server_ok": liveness_ok,
        "config": cfg,
    }

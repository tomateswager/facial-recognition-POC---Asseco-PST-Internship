"""
Webservice dedicado para as rotas DeepFace.
Expoe apenas endpoints de API (sem frontend e sem rotas AWS).
"""
import os
import threading

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.deepface.routes import (
    attributes as attributes_router,
    config as config_router,
    document as document_router,
    identify as identify_router,
    live as live_router,
    models as models_router,
    multi as multi_router,
    quality as quality_router,
    register as register_router,
    verify as verify_router,
    video as video_router,
)


def _parse_cors_origins() -> list[str]:
    raw_origins = os.getenv("DEEPFACE_CORS_ORIGINS", "*")
    origins = [origin.strip() for origin in raw_origins.split(",") if origin.strip()]
    return origins or ["*"]


def _warmup_deepface_models():
    """
    Pre-carrega componentes principais para reduzir cold start.
    """
    try:
        import numpy as np
        from deepface import DeepFace

        blank = np.zeros((112, 112, 3), dtype=np.uint8)
        try:
            DeepFace.represent(
                blank,
                model_name="ArcFace",
                detector_backend="skip",
                enforce_detection=False,
            )
        except Exception:
            pass

        try:
            DeepFace.extract_faces(
                blank,
                detector_backend="yunet",
                enforce_detection=False,
                anti_spoofing=False,
            )
        except Exception:
            pass
        print("[DeepFace API] Warmup ArcFace + YuNet concluido.")
    except Exception as exc:
        print(f"[DeepFace API] Aviso no warmup: {exc}")


app = FastAPI(
    title="DeepFace Webservice",
    description="API dedicada para verificacao, identificacao e registo com DeepFace.",
    version="1.0.0",
)

cors_origins = _parse_cors_origins()
app.add_middleware(
    CORSMiddleware,
    allow_origins=cors_origins,
    allow_credentials="*" not in cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Rotas DeepFace
app.include_router(verify_router.router)
app.include_router(identify_router.router)
app.include_router(register_router.router)
app.include_router(quality_router.router)
app.include_router(models_router.router)
app.include_router(document_router.router)
app.include_router(multi_router.router)
app.include_router(video_router.router)
app.include_router(live_router.router)
app.include_router(attributes_router.router)
app.include_router(config_router.router)


@app.get("/")
async def root():
    return {
        "service": "deepface-webservice",
        "status": "online",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
async def health():
    return {"ok": True, "service": "deepface-webservice"}


@app.on_event("startup")
async def on_startup():
    # Remove sessoes antigas de documento pendentes de execucoes anteriores.
    from backend.deepface.services.document_service import cleanup_old_sessions

    cleanup_old_sessions(max_age_minutes=0)
    threading.Thread(target=_warmup_deepface_models, daemon=True).start()
    print("=" * 60)
    print("  DeepFace Webservice")
    print("=" * 60)
    print("  URL Base:  http://localhost:8001")
    print("  API Docs:  http://localhost:8001/docs")
    print("=" * 60)

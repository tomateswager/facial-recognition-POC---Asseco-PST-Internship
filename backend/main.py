"""
DeepFace POC - Servidor Principal
Prova de Conceito de Reconhecimento Facial para Asseco PST

Funcionalidades:
1. Verificação 1:1 (ficheiro + câmara)
2. Identificação 1:N (ficheiro + câmara)
3. Avaliação Comparativa de Modelos
4. Extração de Foto de Documento de ID (fluxo tipo banco)
5. Anti-Spoofing inteligente (câmara vs ficheiro)
"""
import os as _os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from backend.config import FRONTEND_DIR
from backend.deepface.routes import verify, identify, register, quality, models, document, multi as multi_router, video as video_router, live as live_router, attributes as attributes_router, config as df_config_router
from backend.aws.routes import verify as aws_verify_router, identify as aws_identify_router, register as aws_register_router
from backend.aws.routes import liveness as aws_liveness_router
from backend.aws.routes import config as aws_config_router

app = FastAPI(
    title="DeepFace POC - Reconhecimento Facial",
    description="POC Completo: Verificação 1:1, Identificação 1:N, Comparação de Modelos e Documento de ID",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000", "http://127.0.0.1:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# === Rotas da API — DeepFace ===
app.include_router(verify.router)
app.include_router(identify.router)
app.include_router(register.router)
app.include_router(quality.router)
app.include_router(models.router)
app.include_router(document.router)
app.include_router(multi_router.router)
app.include_router(video_router.router)
app.include_router(live_router.router)
app.include_router(attributes_router.router)
app.include_router(df_config_router.router)

# === Rotas da API — AWS Rekognition ===
app.include_router(aws_verify_router.router)
app.include_router(aws_identify_router.router)
app.include_router(aws_register_router.router)
app.include_router(aws_liveness_router.router)
app.include_router(aws_config_router.router)

# === Servir Frontend ===
app.mount("/css", StaticFiles(directory=f"{FRONTEND_DIR}/css"), name="css")
app.mount("/js", StaticFiles(directory=f"{FRONTEND_DIR}/js"), name="js")
app.mount("/faceapi-models", StaticFiles(directory=f"{FRONTEND_DIR}/models"), name="faceapi-models")

# === Liveness React App ===
_liveness_dist = _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "liveness", "app", "dist")
if _os.path.exists(_liveness_dist):
    app.mount("/liveness", StaticFiles(directory=_liveness_dist, html=True), name="liveness")


# === Páginas HTML — DeepFace ===
@app.get("/")
async def serve_index():
    return FileResponse(f"{FRONTEND_DIR}/index.html")


@app.get("/verify")
async def serve_verify():
    return FileResponse(f"{FRONTEND_DIR}/verify.html")


@app.get("/identify")
async def serve_identify():
    return FileResponse(f"{FRONTEND_DIR}/identify.html")


@app.get("/register")
async def serve_register():
    return FileResponse(f"{FRONTEND_DIR}/register.html")


@app.get("/models")
async def serve_models():
    return FileResponse(f"{FRONTEND_DIR}/models.html")


@app.get("/document")
async def serve_document():
    return FileResponse(f"{FRONTEND_DIR}/document.html")


@app.get("/multi")
async def serve_multi():
    return FileResponse(f"{FRONTEND_DIR}/multi.html")


@app.get("/compare")
async def serve_compare():
    return FileResponse(f"{FRONTEND_DIR}/compare.html")


@app.get("/video")
async def serve_video():
    return FileResponse(f"{FRONTEND_DIR}/video.html")


@app.get("/live")
async def serve_live():
    return FileResponse(f"{FRONTEND_DIR}/live.html")


@app.get("/deepface/config")
async def serve_df_config():
    return FileResponse(f"{FRONTEND_DIR}/deepface-config.html")


# === Páginas HTML — AWS ===
@app.get("/aws/verify")
async def serve_aws_verify():
    return FileResponse(f"{FRONTEND_DIR}/aws-verify.html")


@app.get("/aws/identify")
async def serve_aws_identify():
    return FileResponse(f"{FRONTEND_DIR}/aws-identify.html")


@app.get("/aws")
async def serve_aws_index():
    return FileResponse(f"{FRONTEND_DIR}/aws-index.html")


@app.get("/aws/register")
async def serve_aws_register():
    return FileResponse(f"{FRONTEND_DIR}/aws-register.html")


@app.get("/aws/stats")
async def serve_aws_stats():
    return FileResponse(f"{FRONTEND_DIR}/aws-stats.html")


@app.get("/aws/config")
async def serve_aws_config():
    return FileResponse(f"{FRONTEND_DIR}/aws-config.html")


@app.get("/aws/multi")
async def serve_aws_multi():
    return FileResponse(f"{FRONTEND_DIR}/aws-multi.html")


@app.on_event("startup")
async def startup_event():
    # Limpar sessões de documento antigas (de F5, abandono, etc.)
    from backend.deepface.services.document_service import cleanup_old_sessions
    cleanup_old_sessions(max_age_minutes=0)  # Remove tudo anterior ao arranque

    # Arrancar servidor de liveness em background thread
    import threading
    import requests as _req

    def _start_liveness_server():
        try:
            _req.get("http://localhost:5001/api/health", timeout=1)
            return  # já a correr
        except Exception:
            pass
        try:
            from liveness.server import app as flask_app
            import logging
            logging.getLogger("werkzeug").setLevel(logging.ERROR)
            threading.Thread(target=lambda: flask_app.run(host="127.0.0.1", port=5001, debug=False, use_reloader=False), daemon=True).start()
        except Exception as e:
            print(f"[Liveness] Aviso: servidor não iniciado: {e}")

    _start_liveness_server()

    # Pré-carregar modelo DeepFace em background para eliminar cold start
    def _warmup_deepface():
        try:
            import numpy as np
            from deepface import DeepFace
            blank = np.zeros((112, 112, 3), dtype=np.uint8)
            try:
                DeepFace.represent(blank, model_name="ArcFace", detector_backend="skip", enforce_detection=False)
            except Exception:
                pass
            # Pré-carregar detetor YuNet (usado na câmara ao vivo)
            try:
                DeepFace.extract_faces(blank, detector_backend="yunet", enforce_detection=False, anti_spoofing=False)
            except Exception:
                pass
            print("[Warmup] ArcFace + YuNet pré-carregados.")
        except Exception as e:
            print(f"[Warmup] Aviso: {e}")

    import threading as _threading
    _threading.Thread(target=_warmup_deepface, daemon=True).start()

    print("=" * 60)
    print("  DeepFace POC v2.0 - Reconhecimento Facial")
    print("  Asseco PST - Departamento de Redes e Segurança")
    print("=" * 60)
    print(f"  Modelo:         ArcFace (99.83% LFW)")
    print(f"  Detetor:        RetinaFace")
    print(f"  Anti-Spoofing:  Dinâmico (câmara=ON, ficheiro=OFF)")
    print(f"  Métrica:        Cosine Similarity")
    print("=" * 60)
    print(f"  Dashboard:      http://localhost:8000")
    print(f"  Verificação:    http://localhost:8000/verify")
    print(f"  Identificação:  http://localhost:8000/identify")
    print(f"  Modelos:        http://localhost:8000/models")
    print(f"  Documento ID:   http://localhost:8000/document")
    print(f"  AWS Verify:     http://localhost:8000/aws/verify")
    print(f"  AWS Identify:   http://localhost:8000/aws/identify")
    print(f"  AWS Register:   http://localhost:8000/aws/register")
    print(f"  Liveness:       http://localhost:5001")
    print(f"  API Docs:       http://localhost:8000/docs")
    print("=" * 60)

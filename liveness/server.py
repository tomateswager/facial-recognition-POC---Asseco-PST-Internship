"""
liveness/server.py - Flask API para Amazon Rekognition Face Liveness

Expõe endpoints para o frontend React:
- GET  /api/health       → health check
- GET  /api/credentials  → credenciais AWS temporárias para o componente React
- GET  /api/result/<id>  → resultado da sessão de liveness
- GET  /*                → serve o React app (dist/)
"""

import os
import sys
import json
import time
import base64
import boto3
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
from dotenv import load_dotenv

# Carregar .env do diretório pai
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

DIST_DIR = os.path.join(os.path.dirname(__file__), "app", "dist")
_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "..", "config.json")

app = Flask(__name__, static_folder=DIST_DIR, static_url_path="")

# CORS restrito ao servidor FastAPI (porta 8000) e ao próprio servidor
CORS(app, origins=[
    "http://localhost:8000",
    "http://127.0.0.1:8000",
    "http://localhost:5001",
    "http://127.0.0.1:5001",
])

AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_REGION = os.getenv("AWS_REGION", "eu-west-1")

rekognition = boto3.client(
    "rekognition",
    aws_access_key_id=AWS_ACCESS_KEY,
    aws_secret_access_key=AWS_SECRET_KEY,
    region_name=AWS_REGION,
)

# ─────────────────────────────────────────────────────────────────────────────
# Rate limiting simples (sem dependências externas)
# ─────────────────────────────────────────────────────────────────────────────

_rate_store: dict = {}  # {ip: [timestamps]}
_RATE_LIMIT = 30        # máx. pedidos por janela
_RATE_WINDOW = 60       # janela em segundos


def _check_rate_limit(ip: str) -> bool:
    """Devolve True se o IP ainda está dentro do limite, False se excedeu."""
    now = time.time()
    timestamps = _rate_store.get(ip, [])
    # Remover timestamps fora da janela
    timestamps = [t for t in timestamps if now - t < _RATE_WINDOW]
    if len(timestamps) >= _RATE_LIMIT:
        _rate_store[ip] = timestamps
        return False
    timestamps.append(now)
    _rate_store[ip] = timestamps
    return True


def _get_liveness_threshold() -> float:
    """Lê o threshold de liveness do config.json (sincronizado com app.py)."""
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            cfg = json.load(f)
        return float(cfg.get("liveness_threshold", 75.0))
    except Exception:
        return 75.0


# ─────────────────────────────────────────────────────────────────────────────
# API Endpoints
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/api/health")
def health():
    return jsonify({"status": "ok"})


@app.route("/api/credentials")
def get_credentials():
    """
    Devolve as credenciais AWS para o FaceLivenessDetector.
    NOTA: Em produção, usar AWS Cognito Identity Pool com credenciais temporárias STS.
    Para o POC, devolve as credenciais do .env (apenas acessível em localhost).
    """
    ip = request.remote_addr
    if not _check_rate_limit(ip):
        return jsonify({"error": "Rate limit excedido. Aguarda um momento."}), 429

    return jsonify({
        "accessKeyId": AWS_ACCESS_KEY,
        "secretAccessKey": AWS_SECRET_KEY,
        "region": AWS_REGION,
    })


@app.route("/api/result/<session_id>")
def get_result(session_id):
    """Obtém o resultado da sessão de Face Liveness."""
    ip = request.remote_addr
    if not _check_rate_limit(ip):
        return jsonify({"error": "Rate limit excedido. Aguarda um momento."}), 429

    # Validar session_id (só alfanuméricos e hífenes)
    if not session_id or not all(c.isalnum() or c == '-' for c in session_id):
        return jsonify({"error": "Session ID inválido"}), 400

    try:
        response = rekognition.get_face_liveness_session_results(SessionId=session_id)
        status = response.get("Status", "UNKNOWN")
        confidence = response.get("Confidence", 0.0)

        liveness_threshold = _get_liveness_threshold()
        is_live = status == "SUCCEEDED" and confidence >= liveness_threshold

        result = {
            "status": status,
            "confidence": round(confidence, 2),
            "isLive": is_live,
            "threshold": liveness_threshold,
        }

        # Incluir a imagem de referência capturada durante o desafio
        ref_bytes = response.get("ReferenceImage", {}).get("Bytes")
        if not ref_bytes:
            # Fallback: usar primeira AuditImage se ReferenceImage não disponível
            audit_images = response.get("AuditImages", [])
            if audit_images:
                ref_bytes = audit_images[0].get("Bytes")
        if ref_bytes:
            result["reference_image"] = base64.b64encode(ref_bytes).decode("utf-8")

        return jsonify(result)
    except rekognition.exceptions.SessionNotFoundException:
        return jsonify({"error": "Sessão não encontrada ou expirada"}), 404
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ─────────────────────────────────────────────────────────────────────────────
# Servir o React App
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_react(path):
    if path and os.path.exists(os.path.join(DIST_DIR, path)):
        return send_from_directory(DIST_DIR, path)
    return send_from_directory(DIST_DIR, "index.html")


if __name__ == "__main__":
    port = int(os.getenv("LIVENESS_PORT", 5001))
    print(f"[Face Liveness Server] A iniciar na porta {port}...")
    app.run(host="127.0.0.1", port=port, debug=False, use_reloader=False)

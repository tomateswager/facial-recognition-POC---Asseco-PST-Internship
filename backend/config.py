"""
Configuração central do DeepFace POC
"""
import os

# === Diretórios ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATABASE_DIR = os.path.join(BASE_DIR, "database", "faces")
TEMP_DIR = os.path.join(BASE_DIR, "temp")
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")
DOCUMENTS_DIR = os.path.join(BASE_DIR, "database", "documents")

# === DeepFace - Modelos ===
RECOGNITION_MODEL = "ArcFace"
DETECTOR_BACKEND = "yunet"
DISTANCE_METRIC = "cosine" 

#yunet é o que deteta a cara, ArcFace faz os vetores da cara e cosine 
# calcula a distância entre os vetores

VERIFY_THRESHOLD = 0.54   # 1:1 — mais restritivo (compara duas imagens específicas)
IDENTIFY_THRESHOLD = 0.54  # 1:N — 0.68 default ArcFace (variações de iluminação/ângulo)
CUSTOM_THRESHOLD = IDENTIFY_THRESHOLD  # alias mantido para compatibilidade
# Threshold mais permissivo para comparação documento vs. câmara ao vivo
# (fotos de doc têm iluminação/cor muito diferente da câmara)
DOCUMENT_THRESHOLD = 0.65

# === Modelos para Avaliação Comparativa ===
MODELS_TO_EVALUATE = ["ArcFace", "Facenet512", "VGG-Face", "SFace"]

# === Anti-Spoofing ===
# Nota: anti-spoofing é controlado dinamicamente:
# - Desativado para uploads de ficheiro (iria sempre detetar spoof)
# - Ativado para capturas de câmara em tempo real
ANTI_SPOOFING = False  # Default global (override por pedido)

# === Qualidade de Imagem - Thresholds ===
QUALITY_MIN_FACE_SIZE = 50        # Tamanho mínimo da face em pixels
QUALITY_MAX_BLUR = 30.0           # Variância Laplacian mínima (abaixo = desfocado)
QUALITY_MIN_BRIGHTNESS = 20       # Brilho mínimo médio (0-255)
QUALITY_MAX_BRIGHTNESS = 240      # Brilho máximo médio (0-255)
QUALITY_MIN_FACE_RATIO = 0.02     # Rácio mínimo face/imagem
QUALITY_MAX_YAW_ANGLE = 30        # Ângulo máximo de rotação horizontal

# === Upload ===
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # 10 MB

# === Servidor ===
HOST = "0.0.0.0"
PORT = 8000

# === Configuração Runtime (persiste em config.json) ===
import json as _json

_CONFIG_FILE = os.path.join(BASE_DIR, "config.json")

_DF_DEFAULTS = {
    "df_verify_threshold":      VERIFY_THRESHOLD,
    "df_identify_threshold":    IDENTIFY_THRESHOLD,
    "df_document_threshold":    DOCUMENT_THRESHOLD,
    "df_quality_min_blur":      QUALITY_MAX_BLUR,
    "df_quality_min_face_size": QUALITY_MIN_FACE_SIZE,
}


def get_df_runtime_config() -> dict:
    """Lê thresholds DeepFace do config.json (fallback para defaults de config.py)."""
    cfg = dict(_DF_DEFAULTS)
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            stored = _json.load(f)
        cfg.update({k: stored[k] for k in _DF_DEFAULTS if k in stored})
    except Exception:
        pass
    return cfg


def save_df_runtime_config(updates: dict):
    """Guarda thresholds DeepFace em config.json (merge com valores existentes)."""
    existing = {}
    try:
        with open(_CONFIG_FILE, encoding="utf-8") as f:
            existing = _json.load(f)
    except Exception:
        pass
    existing.update(updates)
    with open(_CONFIG_FILE, "w", encoding="utf-8") as f:
        _json.dump(existing, f, indent=2)

# Criar diretórios se não existirem
os.makedirs(DATABASE_DIR, exist_ok=True)
os.makedirs(TEMP_DIR, exist_ok=True)
os.makedirs(DOCUMENTS_DIR, exist_ok=True)

# === AWS Rekognition ===
from dotenv import load_dotenv
load_dotenv()
AWS_COLLECTION_ID = os.getenv("AWS_COLLECTION_ID", "poc-asseco-faces")
AWS_VERIFY_THRESHOLD = float(os.getenv("AWS_VERIFY_THRESHOLD", "90.0"))
AWS_IDENTIFY_THRESHOLD = float(os.getenv("AWS_IDENTIFY_THRESHOLD", "80.0"))

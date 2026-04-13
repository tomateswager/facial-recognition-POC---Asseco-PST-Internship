"""
Serviço de Avaliação Comparativa de Modelos
Testa e compara modelos de reconhecimento facial disponíveis no DeepFace.
Objetivo 3 do POC.
"""
import os
import time
import traceback
import psutil
import numpy as np
from deepface import DeepFace

from backend.config import (
    MODELS_TO_EVALUATE,
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    TEMP_DIR,
)


def evaluate_models_verify(image1_bytes: bytes, image2_bytes: bytes) -> dict:
    """
    Avalia múltiplos modelos numa verificação 1:1.
    Compara precisão, tempo de processamento e consumo de recursos.

    Args:
        image1_bytes: Bytes da primeira imagem
        image2_bytes: Bytes da segunda imagem

    Returns:
        dict com matriz comparativa de resultados
    """
    import io
    import uuid
    from PIL import Image

    # Guardar imagens temporariamente
    def save_temp(data, name):
        path = os.path.join(TEMP_DIR, f"eval_{name}_{uuid.uuid4().hex}.jpg")
        try:
            img = Image.open(io.BytesIO(data))
            if img.mode != "RGB":
                img = img.convert("RGB")
            img.save(path, "JPEG", quality=95)
        except Exception:
            with open(path, "wb") as f:
                f.write(data)
        return path

    path1 = save_temp(image1_bytes, "img1")
    path2 = save_temp(image2_bytes, "img2")

    results = []

    # Avaliar cada modelo (_evaluate_single_model)
    for model_name in MODELS_TO_EVALUATE:
        model_result = _evaluate_single_model(path1, path2, model_name)
        results.append(model_result)

    # Limpar ficheiros temporários
    for p in [path1, path2]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass

    # Determinar melhor modelo
    successful = [r for r in results if r["success"]]
    best_model = None
    if successful:
        # Critério: menor tempo com verificação correta
        best_model = min(successful, key=lambda r: r["processing_time_ms"])["model"]

    return {
        "success": True,
        "models_tested": len(results),
        "results": results,
        "best_speed": best_model,
        "detector": DETECTOR_BACKEND,
        "metric": DISTANCE_METRIC,
        "message": f"Avaliação completa de {len(results)} modelos.",
    }


def _evaluate_single_model(path1: str, path2: str, model_name: str) -> dict:
    """
    Avalia um modelo individual.
    """
    try:
        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss / 1024 / 1024  # MB

        start_time = time.time()

        result = DeepFace.verify(
            img1_path=path1,
            img2_path=path2,
            model_name=model_name,
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            anti_spoofing=False,
        )

        elapsed_ms = (time.time() - start_time) * 1000
        mem_after = process.memory_info().rss / 1024 / 1024
        mem_used = mem_after - mem_before

        return {
            "model": model_name,
            "success": True,
            "verified": bool(result["verified"]),
            "distance": round(float(result["distance"]), 6),
            "threshold": round(float(result["threshold"]), 6),
            "processing_time_ms": round(elapsed_ms, 1),
            "memory_delta_mb": round(max(0, mem_used), 1),
            "memory_total_mb": round(mem_after, 1),
        }

    except Exception as e:
        return {
            "model": model_name,
            "success": False,
            "error": f"{type(e).__name__}: {str(e)}",
            "processing_time_ms": 0,
            "memory_delta_mb": 0,
            "memory_total_mb": 0,
        }


def get_model_info() -> list:
    """
    Retorna informação teórica sobre os modelos disponíveis.
    """
    return [
        {
            "name": "ArcFace",
            "accuracy_lfw": "99.83%",
            "embedding_size": 512,
            "architecture": "ResNet-100",
            "loss_function": "Additive Angular Margin",
            "notes": "Estado da arte. Melhor precisão geral. Recomendado para produção.",
        },
        {
            "name": "Facenet512",
            "accuracy_lfw": "99.65%",
            "embedding_size": 512,
            "architecture": "Inception-ResNet",
            "loss_function": "Triplet Loss",
            "notes": "Excelente equilíbrio precisão/velocidade. Desenvolvido pelo Google.",
        },
        {
            "name": "VGG-Face",
            "accuracy_lfw": "98.78%",
            "embedding_size": 4096,
            "architecture": "VGG-16",
            "loss_function": "Softmax",
            "notes": "Modelo clássico. Embeddings grandes. Mais lento mas robusto.",
        },
        {
            "name": "SFace",
            "accuracy_lfw": "99.60%",
            "embedding_size": 128,
            "architecture": "MobileNet-based",
            "loss_function": "Sigmoid-Constrained Hypersphere",
            "notes": "Embeddings compactos. Boa opção para dispositivos com recursos limitados.",
        },
    ]

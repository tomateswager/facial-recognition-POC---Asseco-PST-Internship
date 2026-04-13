"""
Serviço DeepFace - Wrapper para reconhecimento facial
Encapsula toda a lógica de verificação 1:1, identificação 1:N e anti-spoofing.

Anti-Spoofing inteligente:
- Upload de ficheiro → anti-spoofing DESATIVADO (deteta sempre spoof em fotos)
- Câmara em tempo real → anti-spoofing ATIVADO

Embeddings:
- Calculados no momento do registo e guardados em embeddings.json por pessoa
- Identificação 1:N usa embeddings pré-calculados (sem reprocessar as imagens)
- Migração automática: pessoas sem embeddings.json têm os vetores calculados na 1ª identificação
"""
import os
import io
import json
import shutil
import uuid
import time
import logging

logger = logging.getLogger(__name__)
import unicodedata
import numpy as np
import cv2
from PIL import Image
from deepface import DeepFace
from backend.config import (
    RECOGNITION_MODEL,
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    DATABASE_DIR,
    TEMP_DIR,
    CUSTOM_THRESHOLD,
    VERIFY_THRESHOLD,
    get_df_runtime_config,
)
_ANTI_SPOOFING_RUNTIME_AVAILABLE = None


_MAX_IMAGE_DIM = 640  # RetinaFace + ArcFace não precisam de mais que isto


def _save_temp_image(file_bytes: bytes, filename: str = None) -> str:
    if filename is None:
        filename = f"{uuid.uuid4().hex}.jpg"
    path = os.path.join(TEMP_DIR, filename)
    try:
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        if max(w, h) > _MAX_IMAGE_DIM:
            ratio = _MAX_IMAGE_DIM / max(w, h)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            logger.debug("[RESIZE] %dx%d → %dx%d", w, h, new_w, new_h)
        img.save(path, "JPEG", quality=95)
    except Exception:
        with open(path, "wb") as f:
            f.write(file_bytes)
    return path


def _cleanup_temp(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _rotate_by_angle(image_bgr: "np.ndarray", angle: int) -> "np.ndarray":
    """Roda uma imagem BGR para um angulo discreto."""
    if angle == 90:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_CLOCKWISE)
    if angle == 270:
        return cv2.rotate(image_bgr, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if angle == 180:
        return cv2.rotate(image_bgr, cv2.ROTATE_180)
    return image_bgr


def _iter_rotations(image_bgr: "np.ndarray"):
    """Gera imagem original e rotacoes candidatas para deteccao facial."""
    return [
        (0, image_bgr),
        (90, _rotate_by_angle(image_bgr, 90)),
        (270, _rotate_by_angle(image_bgr, 270)),
        (180, _rotate_by_angle(image_bgr, 180)),
    ]


def _detect_best_face_orientation(image_path: str) -> dict:
    """
    Procura a melhor orientacao da imagem para deteccao facial.
    Usa backend principal e fallback opencv.
    """
    img = cv2.imread(image_path)
    if img is None:
        return {"found": False}

    backends = [DETECTOR_BACKEND]
    if DETECTOR_BACKEND != "opencv":
        backends.append("opencv")

    best = None
    for backend in backends:
        for angle, rotated in _iter_rotations(img):
            try:
                faces = DeepFace.extract_faces(
                    img_path=rotated,
                    detector_backend=backend,
                    anti_spoofing=False,
                    enforce_detection=False,
                )
            except Exception:
                continue

            for face in faces or []:
                raw_confidence = float(face.get("confidence", 0))
                confidence = 0.72 if backend == "opencv" and raw_confidence <= 0 else raw_confidence
                region = face.get("facial_area", {})
                area = float(region.get("w", 0)) * float(region.get("h", 0))
                better = (
                    best is None
                    or confidence > best["confidence"]
                    or (confidence == best["confidence"] and area > best["area"])
                )
                if better:
                    best = {
                        "found": True,
                        "angle": angle,
                        "backend": backend,
                        "confidence": confidence,
                        "raw_confidence": raw_confidence,
                        "area": area,
                    }
        if best and best["confidence"] >= 0.65:
            break

    return best if best else {"found": False}


def _auto_orient_image_for_face(image_path: str) -> dict:
    """
    Roda a imagem para a orientacao com melhor deteccao de face, se necessario.
    """
    detection = _detect_best_face_orientation(image_path)
    if not detection.get("found"):
        return {"rotated": False, "found": False}

    angle = int(detection.get("angle", 0))
    if angle == 0:
        return {"rotated": False, "found": True, **detection}

    img = cv2.imread(image_path)
    if img is None:
        return {"rotated": False, "found": True, **detection}

    rotated = _rotate_by_angle(img, angle)
    cv2.imwrite(image_path, rotated)
    logger.debug(
        "[AUTO-ROTATE] %s -> %s graus (backend=%s, conf=%.3f)",
        os.path.basename(image_path),
        angle,
        detection.get("backend"),
        float(detection.get("confidence", 0)),
    )
    return {"rotated": True, "found": True, **detection}


def _get_anti_spoofing_runtime_status():
    """
    Verifica se o runtime necessário para anti-spoofing (torch) está disponível.
    """
    global _ANTI_SPOOFING_RUNTIME_AVAILABLE
    if _ANTI_SPOOFING_RUNTIME_AVAILABLE is not None:
        return _ANTI_SPOOFING_RUNTIME_AVAILABLE

    try:
        import torch  # noqa: F401
        _ANTI_SPOOFING_RUNTIME_AVAILABLE = True
    except Exception:
        _ANTI_SPOOFING_RUNTIME_AVAILABLE = False
    return _ANTI_SPOOFING_RUNTIME_AVAILABLE


def _is_anti_spoofing_dependency_error(error_msg: str) -> bool:
    msg = (error_msg or "").lower()
    return (
        "install torch" in msg
        or "no module named 'torch'" in msg
        or "anti spoofing module" in msg
    )


# === Gestão de Embeddings ===

def _embeddings_path(person_dir: str) -> str:
    return os.path.join(person_dir, "embeddings.json")


def _load_person_embeddings(person_dir: str) -> list:
    path = _embeddings_path(person_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_person_embeddings(person_dir: str, embeddings: list):
    path = _embeddings_path(person_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(embeddings, f)


def _compute_distance(a: list, b: list) -> float:
    """Calcula a distância entre dois embeddings de acordo com DISTANCE_METRIC."""
    a_arr = np.array(a, dtype=np.float64)
    b_arr = np.array(b, dtype=np.float64)
    if DISTANCE_METRIC == "cosine":
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        if norm_a == 0 or norm_b == 0:
            return 1.0
        return float(1.0 - np.dot(a_arr, b_arr) / (norm_a * norm_b))
    elif DISTANCE_METRIC == "euclidean_l2":
        norm_a = np.linalg.norm(a_arr)
        norm_b = np.linalg.norm(b_arr)
        a_norm = a_arr / (norm_a + 1e-10)
        b_norm = b_arr / (norm_b + 1e-10)
        return float(np.linalg.norm(a_norm - b_norm))
    else:  # euclidean
        return float(np.linalg.norm(a_arr - b_arr))




def _check_live_anti_spoofing(image_path: str) -> dict:
    """
    Valida prova de vida apenas para a imagem ao vivo.
    Evita usar anti_spoofing diretamente no DeepFace.verify (que valida também img1).
    """
    try:
        faces = DeepFace.extract_faces(
            img_path=image_path,
            detector_backend=DETECTOR_BACKEND,
            anti_spoofing=True,
            enforce_detection=True,
        )
    except ValueError as e:
        error_msg = str(e)
        if _is_anti_spoofing_dependency_error(error_msg):
            return {
                "ok": True,
                "anti_spoofing_available": False,
                "warning": "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada).",
            }
        if "Spoof" in error_msg or "spoof" in error_msg:
            return {
                "ok": False,
                "spoof_detected": True,
                "message": "🚨 Spoofing detetado! A imagem ao vivo não parece ser de uma pessoa real.",
            }
        if "Face" in error_msg:
            return {
                "ok": False,
                "spoof_detected": False,
                "message": "Nenhum rosto detetado na imagem ao vivo.",
            }
        return {"ok": False, "spoof_detected": False, "message": f"Erro: {error_msg}"}
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        if _is_anti_spoofing_dependency_error(error_msg):
            return {
                "ok": True,
                "anti_spoofing_available": False,
                "warning": "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada).",
            }
        return {
            "ok": False,
            "spoof_detected": False,
            "message": f"Erro ao validar prova de vida: {error_msg}",
        }

    if not faces:
        return {"ok": False, "spoof_detected": False, "message": "Nenhum rosto detetado na imagem ao vivo."}

    best_face = max(faces, key=lambda f: f.get("confidence", 0))
    if not best_face.get("is_real", True):
        return {
            "ok": False,
            "spoof_detected": True,
            "message": "🚨 Spoofing detetado! Utilize a câmara com o seu rosto real.",
        }

    return {"ok": True, "anti_spoofing_available": True}


def verify_1to1(image1_bytes: bytes, image2_bytes: bytes, use_anti_spoofing: bool = False) -> dict:
    """
    Verificação 1:1 - Compara dois rostos.

    Args:
        image1_bytes: Bytes da primeira imagem
        image2_bytes: Bytes da segunda imagem
        use_anti_spoofing: Se True, ativa anti-spoofing (câmara). Se False, desativa (ficheiros).
    """
    t0 = time.perf_counter()
    path1 = _save_temp_image(image1_bytes, f"verify1_{uuid.uuid4().hex}.jpg")
    path2 = _save_temp_image(image2_bytes, f"verify2_{uuid.uuid4().hex}.jpg")
    logger.debug(f"[TIMING 1:1] guardar imagens temp: {time.perf_counter()-t0:.3f}s")

    try:
        t_orient = time.perf_counter()
        _auto_orient_image_for_face(path1)
        if not use_anti_spoofing:
            _auto_orient_image_for_face(path2)
        logger.debug(f"[TIMING 1:1] auto-orientacao: {time.perf_counter()-t_orient:.3f}s")

        anti_spoofing_applied = False
        anti_spoofing_warning = None

        if use_anti_spoofing:
            if _get_anti_spoofing_runtime_status():
                t1 = time.perf_counter()
                anti_spoof = _check_live_anti_spoofing(path2)
                logger.debug(f"[TIMING 1:1] anti-spoofing check: {time.perf_counter()-t1:.3f}s")
                if not anti_spoof.get("ok"):
                    return {
                        "success": False,
                        "verified": False,
                        "spoof_detected": anti_spoof.get("spoof_detected", False),
                        "anti_spoofing": True,
                        "message": anti_spoof.get("message", "Falha na validação de prova de vida."),
                    }
                anti_spoofing_applied = anti_spoof.get("anti_spoofing_available", True)
                anti_spoofing_warning = anti_spoof.get("warning")
            else:
                anti_spoofing_warning = "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada)."

        t2 = time.perf_counter()
        verify_backend = DETECTOR_BACKEND
        try:
            result = DeepFace.verify(
                img1_path=path1,
                img2_path=path2,
                model_name=RECOGNITION_MODEL,
                detector_backend=verify_backend,
                distance_metric=DISTANCE_METRIC,
                anti_spoofing=False,
                threshold=get_df_runtime_config()["df_verify_threshold"],
            )
        except ValueError as e:
            if "Face" in str(e) and verify_backend != "opencv":
                verify_backend = "opencv"
                logger.debug("[FALLBACK 1:1] detector_backend -> opencv")
                result = DeepFace.verify(
                    img1_path=path1,
                    img2_path=path2,
                    model_name=RECOGNITION_MODEL,
                    detector_backend=verify_backend,
                    distance_metric=DISTANCE_METRIC,
                    anti_spoofing=False,
                    threshold=get_df_runtime_config()["df_verify_threshold"],
                )
            else:
                raise
        logger.debug(f"[TIMING 1:1] DeepFace.verify: {time.perf_counter()-t2:.3f}s")
        logger.debug(f"[TIMING 1:1] TOTAL: {time.perf_counter()-t0:.3f}s")

        return {
            "success": True,
            "verified": bool(result["verified"]),
            "distance": round(float(result["distance"]), 6),
            "threshold": round(float(result["threshold"]), 6),
            "model": result["model"],
            "similarity_metric": result["similarity_metric"],
            "detector_backend": verify_backend,
            "anti_spoofing": anti_spoofing_applied,
            "anti_spoofing_warning": anti_spoofing_warning,
            "message": "Os rostos pertencem à mesma pessoa." if result["verified"]
                       else "Os rostos NÃO pertencem à mesma pessoa.",
        }

    except ValueError as e:
        error_msg = str(e)
        if "Spoof" in error_msg or "spoof" in error_msg:
            return {
                "success": False,
                "verified": False,
                "spoof_detected": True,
                "message": "🚨 Spoofing detetado! A imagem não parece ser de um rosto real.",
            }
        elif "Face" in error_msg:
            return {
                "success": False,
                "verified": False,
                "message": "Nenhum rosto detetado. Certifique-se de que ambas as imagens contêm rostos visíveis.",
            }
        return {"success": False, "verified": False, "message": f"Erro: {error_msg}"}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "verified": False, "message": f"Erro: {type(e).__name__}: {e}"}

    finally:
        _cleanup_temp(path1, path2)


def identify_1toN(image_bytes: bytes, use_anti_spoofing: bool = False) -> dict:
    """
    Identificação 1:N - Pesquisa um rosto na base de dados usando embeddings pré-calculados.

    Args:
        image_bytes: Bytes da imagem a identificar
        use_anti_spoofing: Se True, ativa anti-spoofing (câmara).
    """
    registered = get_registered_people()
    if not registered:
        return {
            "success": False,
            "results": [],
            "message": "Base de dados vazia. Registe rostos primeiro na secção de Registo.",
        }

    t0 = time.perf_counter()
    path = _save_temp_image(image_bytes, f"identify_{uuid.uuid4().hex}.jpg")
    logger.debug(f"[TIMING 1:N] guardar imagem temp: {time.perf_counter()-t0:.3f}s")

    try:
        t_orient = time.perf_counter()
        _auto_orient_image_for_face(path)
        logger.debug(f"[TIMING 1:N] auto-orientacao: {time.perf_counter()-t_orient:.3f}s")

        anti_spoofing_applied = bool(use_anti_spoofing and _get_anti_spoofing_runtime_status())
        anti_spoofing_warning = None
        if use_anti_spoofing and not anti_spoofing_applied:
            anti_spoofing_warning = "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada)."

        if anti_spoofing_applied:
            t1 = time.perf_counter()
            anti_spoof = _check_live_anti_spoofing(path)
            logger.debug(f"[TIMING 1:N] anti-spoofing check: {time.perf_counter()-t1:.3f}s")
            if not anti_spoof.get("ok"):
                return {
                    "success": False,
                    "results": [],
                    "spoof_detected": anti_spoof.get("spoof_detected", False),
                    "message": anti_spoof.get("message", "Falha na validação de prova de vida."),
                }

        # Calcular embedding da imagem de entrada (1 única vez)
        t2 = time.perf_counter()
        represent_backend = DETECTOR_BACKEND
        try:
            represent_result = DeepFace.represent(
                img_path=path,
                model_name=RECOGNITION_MODEL,
                detector_backend=represent_backend,
                enforce_detection=True,
            )
        except ValueError as e:
            if "Face" in str(e) and represent_backend != "opencv":
                represent_backend = "opencv"
                logger.debug("[FALLBACK 1:N] detector_backend -> opencv")
                represent_result = DeepFace.represent(
                    img_path=path,
                    model_name=RECOGNITION_MODEL,
                    detector_backend=represent_backend,
                    enforce_detection=True,
                )
            else:
                raise
        query_embedding = represent_result[0]["embedding"]
        logger.debug(f"[TIMING 1:N] DeepFace.represent (extrair embedding): {time.perf_counter()-t2:.3f}s")

        # Comparar contra todos os embeddings guardados
        t3 = time.perf_counter()
        all_candidates = []
        for person in registered:
            person_dir = os.path.join(DATABASE_DIR, person["name"])
            embeddings_list = _load_person_embeddings(person_dir)

            for entry in embeddings_list:
                distance = _compute_distance(query_embedding, entry["embedding"])
                all_candidates.append({
                    "name": person["name"],
                    "distance": round(distance, 6),
                })

        all_candidates.sort(key=lambda x: x["distance"])
        _id_threshold = get_df_runtime_config()["df_identify_threshold"]
        results = [c for c in all_candidates if c["distance"] <= _id_threshold]
        logger.debug(f"[TIMING 1:N] comparar embeddings ({len(all_candidates)} candidatos): {time.perf_counter()-t3:.3f}s")
        logger.debug(f"[TIMING 1:N] TOTAL: {time.perf_counter()-t0:.3f}s")

        if results:
            best = results[0]
            message = f"Pessoa identificada: {best['name']} (distância: {best['distance']:.4f})"
        elif all_candidates:
            closest = all_candidates[0]
            message = (
                f"Nenhuma correspondência encontrada na base de dados. "
                f"Candidato mais próximo: {closest['name']} (distância: {closest['distance']:.4f}, "
                f"threshold: {_id_threshold})"
            )
        else:
            message = "Nenhuma correspondência encontrada na base de dados."

        return {
            "success": True,
            "results": results[:5],
            "threshold": _id_threshold,
            "detector_backend": represent_backend,
            "anti_spoofing": anti_spoofing_applied,
            "anti_spoofing_warning": anti_spoofing_warning,
            "total_registered": sum(p["num_images"] for p in registered),
            "message": message,
        }

    except ValueError as e:
        error_msg = str(e)
        if _is_anti_spoofing_dependency_error(error_msg):
            return {
                "success": False,
                "results": [],
                "message": "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada).",
            }
        if "Spoof" in error_msg or "spoof" in error_msg:
            return {
                "success": False,
                "results": [],
                "spoof_detected": True,
                "message": "⚠️ Ataque de spoofing detetado! A imagem não parece ser de um rosto real.",
            }
        elif "Face" in error_msg:
            return {
                "success": False,
                "results": [],
                "message": "Nenhum rosto detetado na imagem.",
            }
        return {"success": False, "results": [], "message": f"Erro: {error_msg}"}

    except Exception as e:
        return {"success": False, "results": [], "message": f"Erro inesperado: {str(e)}"}

    finally:
        _cleanup_temp(path)


def _ascii_safe_name(name: str) -> str:
    """Converte nome para ASCII puro (sem acentos/caracteres especiais) para compatibilidade com DeepFace."""
    normalized = unicodedata.normalize("NFKD", name)
    ascii_str = normalized.encode("ascii", "ignore").decode("ascii")
    return "".join(c for c in ascii_str if c.isalnum() or c in (" ", "-", "_")).strip()


def register_face(image_bytes: bytes, person_name: str, use_anti_spoofing: bool = False) -> dict:
    """
    Registo biométrico - Guarda a imagem e calcula/guarda o embedding do rosto.
    """
    safe_name = _ascii_safe_name(person_name)
    if not safe_name:
        return {"success": False, "message": "Nome inválido."}

    person_dir = os.path.join(DATABASE_DIR, safe_name)
    os.makedirs(person_dir, exist_ok=True)

    temp_path = _save_temp_image(image_bytes, f"register_{uuid.uuid4().hex}.jpg")

    try:
        _auto_orient_image_for_face(temp_path)
        anti_spoofing_applied = bool(use_anti_spoofing and _get_anti_spoofing_runtime_status())

        detection_backend = DETECTOR_BACKEND
        try:
            faces = DeepFace.extract_faces(
                img_path=temp_path,
                detector_backend=detection_backend,
                anti_spoofing=anti_spoofing_applied,
            )
        except ValueError as e:
            if "Face" in str(e) and DETECTOR_BACKEND != "opencv":
                detection_backend = "opencv"
                logger.debug("[FALLBACK REG] detector_backend -> opencv")
                faces = DeepFace.extract_faces(
                    img_path=temp_path,
                    detector_backend=detection_backend,
                    anti_spoofing=anti_spoofing_applied,
                )
            else:
                raise

        if not faces:
            return {"success": False, "message": "Nenhum rosto detetado na imagem."}

        face = faces[0]
        if anti_spoofing_applied and not face.get("is_real", True):
            return {
                "success": False,
                "spoof_detected": True,
                "message": "⚠️ Spoofing detetado! Utilize a câmara com o seu rosto real.",
            }

        # Calcular embedding diretamente da imagem temporária (não guarda a foto)
        embedding_backend = DETECTOR_BACKEND
        try:
            represent_result = DeepFace.represent(
                img_path=temp_path,
                model_name=RECOGNITION_MODEL,
                detector_backend=embedding_backend,
                enforce_detection=True,
            )
        except ValueError as e:
            if "Face" in str(e) and embedding_backend != "opencv":
                embedding_backend = "opencv"
                logger.debug("[FALLBACK REG] represent detector_backend -> opencv")
                represent_result = DeepFace.represent(
                    img_path=temp_path,
                    model_name=RECOGNITION_MODEL,
                    detector_backend=embedding_backend,
                    enforce_detection=True,
                )
            else:
                raise
        embedding = represent_result[0]["embedding"]

        embeddings_list = _load_person_embeddings(person_dir)
        embeddings_list.append({"embedding": embedding})
        _save_person_embeddings(person_dir, embeddings_list)

        num_total = len(embeddings_list)

        return {
            "success": True,
            "name": safe_name,
            "num_images": num_total,
            "anti_spoofing": anti_spoofing_applied,
            "anti_spoofing_warning": (
                "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada)."
                if use_anti_spoofing and not anti_spoofing_applied
                else None
            ),
            "message": f"Rosto de '{safe_name}' registado com sucesso! ({num_total} imagem(ns) total)",
        }

    except ValueError as e:
        error_msg = str(e)
        if _is_anti_spoofing_dependency_error(error_msg):
            return {
                "success": False,
                "message": "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada).",
            }
        if "Spoof" in error_msg or "spoof" in error_msg:
            return {
                "success": False,
                "spoof_detected": True,
                "message": "⚠️ Spoofing detetado! Utilize uma imagem real.",
            }
        return {"success": False, "message": f"Erro: {error_msg}"}

    except Exception as e:
        return {"success": False, "message": f"Erro inesperado: {str(e)}"}

    finally:
        _cleanup_temp(temp_path)


def get_registered_people() -> list:
    people = []
    if os.path.exists(DATABASE_DIR):
        for name in sorted(os.listdir(DATABASE_DIR)):
            person_dir = os.path.join(DATABASE_DIR, name)
            if os.path.isdir(person_dir):
                embeddings = _load_person_embeddings(person_dir)
                if embeddings:
                    people.append({"name": name, "num_images": len(embeddings)})
    return people


def delete_person(person_name: str) -> dict:
    db_real = os.path.realpath(DATABASE_DIR)
    person_dir = os.path.realpath(os.path.join(DATABASE_DIR, person_name))
    if not person_dir.startswith(db_real + os.sep):
        return {"success": False, "message": "Nome inválido."}
    if os.path.exists(person_dir) and os.path.isdir(person_dir):
        shutil.rmtree(person_dir)
        return {"success": True, "message": f"'{person_name}' removido(a) da base de dados."}
    return {"success": False, "message": f"Pessoa '{person_name}' não encontrada."}


_EMOTION_EMOJI = {
    "happy": "😊", "sad": "😢", "angry": "😠", "fear": "😨",
    "surprise": "😲", "disgust": "🤢", "neutral": "😐",
}


_AGE_MARGIN = 4  # margem típica de erro do modelo de idade DeepFace


def analyze_face_attributes(image_bytes: bytes) -> dict:
    """Analisa atributos faciais: idade, género, emoção e etnia."""
    temp_path = _save_temp_image(image_bytes)
    try:
        results = DeepFace.analyze(
            img_path=temp_path,
            actions=["age", "gender", "emotion", "race"],
            detector_backend="yunet",
            enforce_detection=False,
            silent=True,
        )
    except Exception as e:
        return {"success": False, "message": str(e)}
    finally:
        try:
            os.remove(temp_path)
        except Exception:
            pass

    if not results:
        return {"success": False, "message": "Nenhuma face detetada"}

    r = results[0] if isinstance(results, list) else results

    # Idade com intervalo de confiança
    age_est = int(round(float(r.get("age") or 0)))
    age_range = {"low": max(0, age_est - _AGE_MARGIN), "high": age_est + _AGE_MARGIN}

    # Género com confiança
    gender_raw = r.get("gender") or {}
    dominant_gender = r.get("dominant_gender", "Unknown")
    gender_conf = round(float(gender_raw.get(dominant_gender, 0)), 1)

    # Emoção
    dominant_emotion = (r.get("dominant_emotion") or "neutral").lower()
    emotions_raw = r.get("emotion") or {}
    sorted_emotions = sorted(emotions_raw.items(), key=lambda x: float(x[1]), reverse=True)

    # Etnia
    dominant_race = (r.get("dominant_race") or "").lower()
    race_raw = r.get("race") or {}
    sorted_race = sorted(race_raw.items(), key=lambda x: float(x[1]), reverse=True)

    return {
        "success": True,
        "age": age_est,
        "age_range": age_range,
        "gender": dominant_gender,
        "gender_confidence": gender_conf,
        "dominant_emotion": dominant_emotion,
        "emotion_emoji": _EMOTION_EMOJI.get(dominant_emotion, "😐"),
        "emotions": [{"name": k.lower(), "confidence": round(float(v), 1)} for k, v in sorted_emotions[:5]],
        "dominant_race": dominant_race,
        "race": [{"name": k.lower(), "confidence": round(float(v), 1)} for k, v in sorted_race[:4]],
    }


def get_system_info() -> dict:
    return {
        "recognition_model": RECOGNITION_MODEL,
        "detector_backend": DETECTOR_BACKEND,
        "distance_metric": DISTANCE_METRIC,
        "custom_threshold": CUSTOM_THRESHOLD,
        "anti_spoofing_note": "Dinâmico: ativo para câmara, desativado para ficheiros",
        "database_path": DATABASE_DIR,
        "registered_people": len(get_registered_people()),
        "total_images": sum(p["num_images"] for p in get_registered_people()),
    }

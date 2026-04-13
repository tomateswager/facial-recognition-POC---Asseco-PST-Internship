"""
Rota de Pesquisa em Vídeo
POST /api/video/search — extrai frames do vídeo e pesquisa uma face (SSE streaming)

Qualidade de frame:
- Selecção do frame mais nítido numa janela de frames adjacentes
- Unsharp mask aplicado antes da comparação
- Skip de frames demasiado desfocados (Laplacian < MIN_SHARPNESS)
- Deduplicação de matches próximos (< DEDUP_WINDOW_S segundos)
"""
import io
import os
import json
import base64
import math
import tempfile
import time
from typing import Optional

import cv2
import numpy as np
from PIL import Image
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import StreamingResponse

from backend.deepface.services.deepface_service import verify_1to1, identify_1toN

router = APIRouter(prefix="/api/video", tags=["Vídeo"])

_MAX_DURATION_S  = 300    # 5 minutos máx
_MIN_SHARPNESS   = 40.0   # Laplacian variance mínima para tentar comparar
_SHARP_WINDOW    = 4      # nº de frames adjacentes a comparar para escolher o mais nítido
_DEDUP_WINDOW_S  = 2.0    # segundos mínimos entre dois matches reportados (evita duplicados)


# ── Utilitários de qualidade ───────────────────────────────────────────────

def _laplacian_sharpness(frame_bgr: np.ndarray) -> float:
    """Variância do Laplaciano — maior = mais nítido."""
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _sharpen(frame_bgr: np.ndarray, strength: float = 1.4) -> np.ndarray:
    """Unsharp mask: realça bordas para melhorar faces ligeiramente desfocadas."""
    blurred = cv2.GaussianBlur(frame_bgr, (0, 0), 2.5)
    return cv2.addWeighted(frame_bgr, strength, blurred, -(strength - 1.0), 0)


def _best_frame_in_window(cap: cv2.VideoCapture, base_idx: int,
                           window: int, fps: float) -> tuple:
    """
    Lê `window` frames a partir de base_idx e devolve o mais nítido.
    Restaura a posição do cap para base_idx + 1 depois.
    Devolve (frame_bgr, sharpness).
    """
    saved_pos = cap.get(cv2.CAP_PROP_POS_FRAMES)
    candidates = []

    for offset in range(window):
        cap.set(cv2.CAP_PROP_POS_FRAMES, base_idx + offset)
        ret, f = cap.read()
        if not ret:
            break
        candidates.append((f, _laplacian_sharpness(f)))

    # Restaurar posição
    cap.set(cv2.CAP_PROP_POS_FRAMES, saved_pos)

    if not candidates:
        return None, 0.0
    return max(candidates, key=lambda x: x[1])


# ── Multi-face: deteção de todas as faces num frame ────────────────────────

def _extract_face_crops(frame_bgr: np.ndarray, min_face_px: int = 50) -> list:
    """
    Deteta todas as faces no frame com YuNet.
    Devolve lista de (face_crop_bytes, (x, y, w, h)).
    Lista vazia = nenhuma face detetada.
    """
    from deepface import DeepFace as _DF
    try:
        faces = _DF.extract_faces(
            img_path=frame_bgr,
            detector_backend="yunet",
            enforce_detection=False,
            anti_spoofing=False,
        )
    except Exception:
        return []

    crops = []
    img_h, img_w = frame_bgr.shape[:2]
    for face in (faces or []):
        region = face.get("facial_area", {})
        x = int(region.get("x", 0))
        y = int(region.get("y", 0))
        w = int(region.get("w", 0))
        h = int(region.get("h", 0))
        if w < min_face_px or h < min_face_px:
            continue
        # margem de 25% para dar contexto à face
        mx, my = int(w * 0.25), int(h * 0.25)
        x1 = max(0, x - mx);  y1 = max(0, y - my)
        x2 = min(img_w, x + w + mx);  y2 = min(img_h, y + h + my)
        crop_bgr = frame_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        buf = io.BytesIO()
        Image.fromarray(rgb).save(buf, format="JPEG", quality=90)
        crops.append((buf.getvalue(), (x, y, w, h)))
    return crops


# ── Conversões ─────────────────────────────────────────────────────────────

def _frame_to_bytes(frame_bgr: np.ndarray, max_dim: int = 640, quality: int = 88) -> bytes:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    h, w = frame_rgb.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        frame_rgb = cv2.resize(frame_rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    buf = io.BytesIO()
    Image.fromarray(frame_rgb).save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def _frame_to_thumb_b64(frame_bgr: np.ndarray, size: int = 220) -> str:
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img.thumbnail((size, size))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=72)
    return base64.b64encode(buf.getvalue()).decode()


def _fmt_ts(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m:02d}:{s:02d}"


# ── Rota principal ─────────────────────────────────────────────────────────

@router.post("/search")
async def video_search(
    video: UploadFile = File(...),
    mode: str = Form("1:1"),
    reference: Optional[UploadFile] = File(None),
    fps_sample: float = Form(1.0),
):
    video_bytes = await video.read()
    ref_bytes   = (await reference.read()) if reference and reference.filename else None

    async def generate():
        tmp_path = None
        try:
            suffix   = os.path.splitext(video.filename or "video.mp4")[1] or ".mp4"
            tmp      = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
            tmp.write(video_bytes)
            tmp.close()
            tmp_path = tmp.name

            cap = cv2.VideoCapture(tmp_path)
            if not cap.isOpened():
                yield f"data: {json.dumps({'type': 'error', 'message': 'Não foi possível abrir o vídeo.'})}\n\n"
                return

            fps          = cap.get(cv2.CAP_PROP_FPS) or 25.0
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration     = total_frames / fps

            if duration > _MAX_DURATION_S:
                yield f"data: {json.dumps({'type': 'error', 'message': f'Vídeo demasiado longo (máx {_MAX_DURATION_S//60} min).'})}\n\n"
                cap.release()
                return

            sample_every     = max(1, int(fps / fps_sample))
            # Contar corretamente os frames amostrados, incluindo o frame 0.
            total_to_process = max(1, math.ceil(total_frames / sample_every))

            yield f"data: {json.dumps({'type': 'info', 'duration': round(duration, 1), 'fps': round(fps, 1), 'total_frames': total_to_process, 'mode': mode})}\n\n"

            matches                  = []
            frame_idx                = 0
            processed                = 0
            skipped_blur             = 0
            last_match_ts            = -_DEDUP_WINDOW_S  # dedup 1:1
            last_match_ts_per_person = {}               # dedup 1:N por pessoa
            t_start                  = time.perf_counter()

            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_idx % sample_every == 0:
                    timestamp = frame_idx / fps
                    processed += 1

                    yield f"data: {json.dumps({'type': 'progress', 'frame': processed, 'total': total_to_process, 'timestamp': round(timestamp, 1), 'ts_fmt': _fmt_ts(timestamp), 'skipped_blur': skipped_blur})}\n\n"

                    try:
                        # 1. Escolher o frame mais nítido na janela
                        best_frame, sharpness = _best_frame_in_window(
                            cap, frame_idx, _SHARP_WINDOW, fps
                        )
                        if best_frame is None:
                            best_frame, sharpness = frame, _laplacian_sharpness(frame)

                        # 2. Skip se demasiado desfocado
                        if sharpness < _MIN_SHARPNESS:
                            skipped_blur += 1
                            frame_idx += 1
                            continue

                        # 3. Aplicar unsharp mask se nitidez moderada
                        if sharpness < 120.0:
                            best_frame = _sharpen(best_frame)

                        # 4. Para 1:1: dedup global (procura uma pessoa específica)
                        if mode == "1:1" and (timestamp - last_match_ts) < _DEDUP_WINDOW_S:
                            frame_idx += 1
                            continue

                        # 5. Detetar TODAS as faces no frame
                        face_crops = _extract_face_crops(best_frame)
                        if not face_crops:
                            frame_idx += 1
                            continue

                        thumb_b64 = _frame_to_thumb_b64(best_frame)

                        for face_bytes, _bbox in face_crops:
                            try:
                                if mode == "1:1":
                                    if not ref_bytes:
                                        break
                                    result = verify_1to1(ref_bytes, face_bytes, use_anti_spoofing=False)
                                    if result.get("verified"):
                                        last_match_ts = timestamp
                                        match = {
                                            "type":         "match",
                                            "timestamp":    round(timestamp, 1),
                                            "ts_fmt":       _fmt_ts(timestamp),
                                            "frame_number": frame_idx,
                                            "thumbnail":    thumb_b64,
                                            "similarity":   round((1 - result.get("distance", 1)) * 100, 1),
                                            "identity":     None,
                                            "sharpness":    round(sharpness, 1),
                                        }
                                        matches.append(match)
                                        yield f"data: {json.dumps(match)}\n\n"
                                        break  # encontrou a pessoa, passa ao próximo frame
                                else:  # 1:N — identificar cada face independentemente
                                    result = identify_1toN(face_bytes, use_anti_spoofing=False)
                                    if result.get("success") and result.get("results"):
                                        best_r = result["results"][0]
                                        person_name = best_r.get("name", "—")
                                        last_ts = last_match_ts_per_person.get(person_name, -_DEDUP_WINDOW_S)
                                        if (timestamp - last_ts) >= _DEDUP_WINDOW_S:
                                            last_match_ts_per_person[person_name] = timestamp
                                            match = {
                                                "type":         "match",
                                                "timestamp":    round(timestamp, 1),
                                                "ts_fmt":       _fmt_ts(timestamp),
                                                "frame_number": frame_idx,
                                                "thumbnail":    thumb_b64,
                                                "similarity":   round((1 - best_r.get("distance", 1)) * 100, 1),
                                                "identity":     person_name,
                                                "sharpness":    round(sharpness, 1),
                                            }
                                            matches.append(match)
                                            yield f"data: {json.dumps(match)}\n\n"
                            except Exception:
                                pass

                    except Exception:
                        pass  # frame sem face detectável — continuar

                frame_idx += 1

            cap.release()
            elapsed = round((time.perf_counter() - t_start) * 1000)
            yield f"data: {json.dumps({'type': 'done', 'match_count': len(matches), 'frames_processed': processed, 'skipped_blur': skipped_blur, 'duration': round(duration, 1), 'elapsed_ms': elapsed})}\n\n"

        finally:
            if tmp_path:
                try:
                    os.unlink(tmp_path)
                except Exception:
                    pass

    return StreamingResponse(generate(), media_type="text/event-stream")

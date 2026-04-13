"""
Serviço de Extração de Fotografia de Documento de Identificação
Extrai e valida a fotografia do titular a partir de documentos de identificação.
Objetivo 5 do POC.

Fluxo:
1. Upload do documento (CC, Passaporte, Carta de Condução)
2. Deteção e recorte da face no documento
3. Validação de qualidade da face extraída
4. Captura em tempo real via câmara
5. Verificação 1:1 (face do documento vs face real)
"""
import os
import io
import time
import uuid
import shutil
import numpy as np
from PIL import Image, ImageEnhance
from deepface import DeepFace

from backend.config import (
    DETECTOR_BACKEND,
    DISTANCE_METRIC,
    DOCUMENTS_DIR,
    TEMP_DIR,
    DOCUMENT_THRESHOLD,
    get_df_runtime_config,
)
from backend.deepface.services.quality_service import full_quality_assessment
import logging
import cv2

logger = logging.getLogger(__name__)

def _preprocess_face_for_comparison(img_bgr: "np.ndarray") -> "np.ndarray":
    """
    Normaliza uma imagem de face para melhorar a comparação documento vs. câmara.

    Problema: fotos de documentos (CC, Passaporte) são muitas vezes a P&B, com
    iluminação de estúdio e comprimidas. A câmara capta a cores com iluminação
    ambiente. Esta diferença faz com que o DeepFace reporta distâncias maiores.

    Solução:
    - Deteta automaticamente se a imagem é a preto e branco
    - Aplica CLAHE (equalização adaptativa de histograma) para normalizar
      iluminação e contraste em ambas as imagens
    - Imagens a P&B: CLAHE no canal cinzento e conversão para 3 canais
    - Imagens a cores: CLAHE no canal L do espaço LAB (preserva cor)
    """

    # Detetar se a imagem é efetivamente a preto e branco
    # (muitos documentos têm foto em RGB mas com canais quase idênticos)
    if len(img_bgr.shape) == 2:
        is_grayscale = True
        gray = img_bgr
    else:
        b, g, r = img_bgr[:, :, 0], img_bgr[:, :, 1], img_bgr[:, :, 2]
        diff_rg = float(np.mean(np.abs(r.astype(np.float32) - g.astype(np.float32))))
        diff_rb = float(np.mean(np.abs(r.astype(np.float32) - b.astype(np.float32))))
        is_grayscale = diff_rg < 12 and diff_rb < 12
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    if is_grayscale:
        equalized = clahe.apply(gray)
        # Converter de volta para 3 canais BGR para o DeepFace processar normalmente
        return cv2.cvtColor(equalized, cv2.COLOR_GRAY2BGR)
    else:
        # Aplicar CLAHE apenas no canal L (luminosidade) preservando a cor
        lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        l_eq = clahe.apply(l_ch)
        lab_eq = cv2.merge([l_eq, a_ch, b_ch])
        return cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)


def _save_temp(file_bytes: bytes, prefix: str = "doc", max_dim: int = None) -> str:
    """Guarda imagem temporária, com redimensionamento opcional."""
    filename = f"{prefix}_{uuid.uuid4().hex}.jpg"
    path = os.path.join(TEMP_DIR, filename)
    try:
        img = Image.open(io.BytesIO(file_bytes))
        if img.mode != "RGB":
            img = img.convert("RGB")
        if max_dim:
            w, h = img.size
            if max(w, h) > max_dim:
                ratio = max_dim / max(w, h)
                new_w, new_h = int(w * ratio), int(h * ratio)
                img = img.resize((new_w, new_h), Image.LANCZOS)
                logger.debug(f"[RESIZE {prefix}] {w}x{h} → {new_w}x{new_h}")
        img.save(path, "JPEG", quality=95)
    except Exception:
        with open(path, "wb") as f:
            f.write(file_bytes)
    return path


def _cleanup(*paths):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _detect_document_rectangle(img_bgr: np.ndarray) -> bool:
    """
    Verifica se existe um retângulo proeminente na imagem (o documento em si).

    Métodos:
    1. Pré-filtro de densidade de arestas: screenshots de UI têm muito mais arestas
       estruturais do que uma foto de um documento físico → rejeita screenshots cedo.
    2. Otsu (ambas as polaridades): trata o documento como região preenchida.
    3. Canny sensível + dilatação forte: deteta CC em fundos de baixo contraste
       (ex. cartão azul claro sobre mesa branca) onde o Otsu falha.

    Em todos os casos, o contorno do documento deve ocupar 10-85% da imagem
    (evita rejeitar documentos pequenos e evita aceitar screenshots que cobrem a imagem toda).
    """

    h, w = img_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        small = img_bgr.copy()

    sh, sw = small.shape[:2]
    image_area = sh * sw
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7, 7), 0)

    # Pré-filtro: densidade de arestas estruturais.
    # Screenshots de UI têm muitas arestas (bordas de painéis, texto, ícones).
    # Fotos de documentos físicos têm grandes áreas uniformes → densidade baixa.
    edges_pre = cv2.Canny(blur, 30, 100)
    edge_density = np.count_nonzero(edges_pre) / image_area
    if edge_density > 0.12:
        return False

    def find_doc_rect(binary):
        """Procura um contorno retangular de tamanho de documento no mapa binário."""
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return False
        for cnt in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
            area = cv2.contourArea(cnt)
            if area < image_area * 0.10:
                break
            if area > image_area * 0.93:  # cobre >93% → conteúdo enche a imagem toda
                continue
            rect = cv2.minAreaRect(cnt)
            rw, rh = rect[1]
            if min(rw, rh) < 10:
                continue
            fill_ratio = area / (rw * rh + 1e-6)
            # fill_ratio adaptativo:
            # - Blob grande (>55% imagem): exige ≥0.80 — o blob do fundo escuro de um
            #   screenshot ocupa ~70% mas tem fill_ratio ~0.68 (tem buracos de imagens inline)
            #   enquanto um CC real fotografado de perto é muito retangular (fill_ratio ~0.92+).
            # - Blob menor (≤55% imagem): basta ≥0.70 — cartão com algum fundo à volta.
            min_fill = 0.80 if area > image_area * 0.55 else 0.70
            if fill_ratio > min_fill:
                ar = max(rw, rh) / (min(rw, rh) + 1e-6)
                # 1.3-3.5: exclui imagens quadradas (fotos inline, ar≈1.0)
                # e colunas muito altas (sidebar com avatares, ar>4).
                # CC ≈ 1.58, Passaporte ≈ 1.42, Carta de Condução ≈ 1.58 → todos dentro.
                if 1.3 <= ar <= 3.5:
                    return True
        return False

    # Método 1: Otsu — robusto para documentos com bom contraste com o fundo
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    _, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    for polarity in [thresh, cv2.bitwise_not(thresh)]:
        closed = cv2.morphologyEx(polarity, cv2.MORPH_CLOSE, kernel, iterations=4)
        if find_doc_rect(closed):
            return True

    # Método 2: canal de saturação (HSV) — para documentos coloridos em fundo neutro.
    # Um CC azul claro tem saturação ≈ 50-80; a mesa branca tem saturação ≈ 0.
    # Ao contrário do cinzento (onde as luminosidades são parecidas), a saturação
    # distingue claramente o cartão do fundo mesmo quando o Canny falha.
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    sat = hsv[:, :, 1]
    _, sat_thresh = cv2.threshold(sat, 15, 255, cv2.THRESH_BINARY)
    kernel_big = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    sat_closed = cv2.morphologyEx(sat_thresh, cv2.MORPH_CLOSE, kernel_big, iterations=5)
    if find_doc_rect(sat_closed):
        return True

    return False


def _has_text_regions(img_bgr: np.ndarray) -> bool:
    """
    Deteta se a imagem contém regiões de texto (caracteres típicos de um documento).
    Usa threshold adaptativo + análise de componentes conectados de tamanho de caractere.

    Documentos têm dezenas/centenas de caracteres visíveis.
    Selfies têm poucos ou nenhum componente nesse intervalo de tamanho.
    """

    h, w = img_bgr.shape[:2]
    max_dim = 800
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        small = cv2.resize(img_bgr, (int(w * scale), int(h * scale)))
    else:
        small = img_bgr.copy()

    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)

    # Threshold adaptativo: encontra texto escuro em fundo claro (documentos típicos)
    thresh = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2
    )

    # Tamanho típico de caracteres a 800px: 5-30px altura, 3-25px largura, área 25-500px²
    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(thresh, connectivity=8)

    text_like = 0
    for i in range(1, num_labels):
        cw, ch, area = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT], stats[i, cv2.CC_STAT_AREA]
        if 25 <= area <= 500 and 5 <= ch <= 30 and 3 <= cw <= 25:
            text_like += 1

    return text_like > 60  # documentos têm facilmente 100+; selfies têm <30


def validate_document_image(document_bytes: bytes) -> dict:
    """
    Validação rápida de uma imagem antes da extração completa.
    Verifica resolução, orientação (selfie vs documento) e tenta detetar face.

    Checks:
    - Resolução mínima 300×200 px (bloqueante)
    - Orientação: ratio < 0.75 → provavelmente selfie (bloqueante)
    - Presença de face (informativo — não bloqueia, apenas avisa)

    Returns:
        valid (bool): False se algum check bloqueante falhou
        has_face (bool): se uma face foi detetada no documento
        checks (list): detalhe de cada check
    """

    nparr = np.frombuffer(document_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None:
        return {
            "valid": False,
            "has_face": False,
            "checks": [{"label": "Formato", "message": "Ficheiro inválido ou corrompido.", "passed": False, "blocking": True}],
            "message": "Não foi possível ler a imagem.",
        }

    h, w = img.shape[:2]
    checks = []

    # 1. Resolução mínima
    resolution_ok = w >= 300 and h >= 200
    checks.append({
        "label": "Resolução",
        "message": f"Resolução demasiado baixa ({w}×{h}px) — tire uma foto com melhor qualidade"
                   if not resolution_ok else "Resolução adequada",
        "passed": resolution_ok,
        "blocking": True,
    })

    # 2. Orientação — aviso informativo, não bloqueia (documentos podem estar rodados 90°)
    ratio = (w / h) if h > 0 else 1.0
    is_portrait = ratio < 0.75
    checks.append({
        "label": "Orientação",
        "message": "Imagem em modo retrato — documentos devem estar na horizontal (pode ser um documento rodado)"
                   if is_portrait else "Orientação correta",
        "passed": not is_portrait,
        "blocking": False,
    })

    hard_ok = resolution_ok  # só a resolução bloqueia; orientação é aviso

    has_face = False
    has_rectangle = False
    has_text = False
    face_ratio = 0.0

    if hard_ok:
        # 3. Deteção de cantos do documento (retângulo)
        has_rectangle = _detect_document_rectangle(img)
        checks.append({
            "label": "Cantos do Documento",
            "message": "Bordas do documento detetadas" if has_rectangle
                       else "Bordas do documento não detetadas",
            "passed": has_rectangle,
            "blocking": False,
        })

        # 4. Deteção de texto (caracteres típicos de documento)
        has_text = _has_text_regions(img)
        checks.append({
            "label": "Texto no Documento",
            "message": "Texto detetado no documento" if has_text
                       else "Texto não detetado — verifique se o documento está legível",
            "passed": has_text,
            "blocking": False,
        })

        # 5. Deteção de face + rácio face/imagem
        try:
            faces = DeepFace.extract_faces(
                img_path=img,
                detector_backend=DETECTOR_BACKEND,
                anti_spoofing=False,
                enforce_detection=False,
            )
            if faces:
                best_face = max(faces, key=lambda f: f.get("confidence", 0))
                if best_face.get("confidence", 0) > 0.4:
                    has_face = True
                    region = best_face.get("facial_area", {})
                    fw, fh = region.get("w", 0), region.get("h", 0)
                    face_ratio = (fw * fh) / (w * h) if w * h > 0 else 0.0
        except Exception:
            pass

        # Face pequena = no documento (< 20%); face grande = selfie (> 20%)
        is_small_face = has_face and face_ratio < 0.12
        checks.append({
            "label": "Face no Documento",
            "message": "Face detetada no documento" if is_small_face
                       else ("Face ocupa demasiado espaço — parece uma selfie" if has_face
                             else "Face não detetada — verifique se o documento está legível"),
            "passed": is_small_face,
            "blocking": False,
        })

    is_small_face = has_face and face_ratio < 0.12
    has_large_face = has_face and not is_small_face  # face dominante → selfie

    # Documento confirmado: texto + retângulo físico detetado + sem face grande.
    # Exigir sempre has_rectangle evita falsos positivos de screenshots com UI complexa.
    # has_large_face bloqueia selfies (face ocupa >12% da imagem).
    # Cobre CC normal (face pequena detetada), CC rodado (face não detetada pelo YuNet)
    # e passaporte (face pequena ou não detetada dependendo da resolução).
    has_document = has_text and has_rectangle and not has_large_face

    failed_blocking = [c for c in checks if not c["passed"] and c.get("blocking")]
    if failed_blocking:
        message = "; ".join(c["message"] for c in failed_blocking)
    elif has_document:
        message = "Documento válido"
    elif has_large_face:
        message = "Parece uma selfie — carregue a foto do documento (CC, Passaporte, etc.)"
    elif has_face and is_small_face:
        message = "Face detetada mas sem texto nem bordas — carregue o documento completo"
    else:
        message = "Não foi possível confirmar o documento — certifique-se que é uma foto clara do CC/Passaporte"

    return {
        "valid": hard_ok,
        "has_face": has_face,
        "has_rectangle": has_rectangle,
        "has_text": has_text,
        "has_document": has_document,
        "checks": checks,
        "message": message,
    }


def extract_face_from_document(document_bytes: bytes) -> dict:
    """
    Extrai a fotografia do titular a partir de uma imagem de documento de ID.

    Passos:
    1. Detetar face no documento
    2. Recortar (crop) com margem
    3. Melhorar qualidade se necessário (nitidez, contraste)
    4. Validar qualidade da face extraída
    5. Retornar face recortada + métricas

    Args:
        document_bytes: Bytes da imagem do documento

    Returns:
        dict com face extraída (base64), métricas de qualidade, session_id
    """
    import base64

    from concurrent.futures import ThreadPoolExecutor

    doc_path = _save_temp(document_bytes, "doc_input")

    # Iniciar OCR em paralelo com a extração de face — ambos precisam apenas
    # dos bytes originais e são independentes entre si.
    executor = ThreadPoolExecutor(max_workers=1)
    ocr_future = executor.submit(extract_document_ocr_data, document_bytes)

    try:
        # Ler imagem
        img_np = cv2.imread(doc_path)
        if img_np is None:
            return {"success": False, "message": "Não foi possível ler a imagem do documento."}

        h, w = img_np.shape[:2]

        # Verificar resolução mínima
        if w < 300 or h < 200:
            return {"success": False, "message": "Resolução do documento muito baixa. Tire uma foto com melhor qualidade."}

        # Pré-processar com CLAHE para melhorar o contraste antes da deteção.
        # Documentos antigos (Carta de Condução, Passaporte) têm fotos a P&B com baixo
        # contraste que fazem o YuNet retornar confiança muito baixa.
        # CLAHE na luminosidade melhora o contraste local sem alterar as dimensões.
        detect_path = doc_path
        try:
            base_p, ext_p = os.path.splitext(doc_path)
            enhanced_path = base_p + "_det" + ext_p
            gray_det = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
            clahe_det = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            enhanced = clahe_det.apply(gray_det)
            img_enh = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
            cv2.imwrite(enhanced_path, img_enh)
            detect_path = enhanced_path
        except Exception:
            pass

        # Tentar detetar com YuNet; se confiança insuficiente, tentar "opencv"
        # (mais tolerante com fotos pequenas/P&B/de baixa qualidade)
        faces = []
        for backend in [DETECTOR_BACKEND, "opencv"]:
            try:
                candidates = DeepFace.extract_faces(
                    img_path=detect_path,
                    detector_backend=backend,
                    anti_spoofing=False,
                    enforce_detection=False,
                )
                # Threshold 0.15: mais permissivo para fotos de documento (P&B, antigas)
                candidates = [f for f in (candidates or []) if f.get("confidence", 0) > 0.15]
                if candidates:
                    faces = candidates
                    break
            except Exception:
                continue

        # Limpar imagem de deteção temporária
        if detect_path != doc_path:
            _cleanup(detect_path)

        if not faces:
            return {
                "success": False,
                "message": "Nenhuma face detetada no documento. Certifique-se de que a foto do documento está bem iluminada e visível.",
            }

        # Usar a face com maior confiança
        best_face = max(faces, key=lambda f: f.get("confidence", 0))
        region = best_face.get("facial_area", {})

        fx, fy = region.get("x", 0), region.get("y", 0)
        fw, fh = region.get("w", 0), region.get("h", 0)

        if fw < 30 or fh < 30:
            return {"success": False, "message": "Face detetada é demasiado pequena. Aproxime o documento da câmara."}

        # Recortar com margem de 30%
        margin = 0.3
        x1 = max(0, int(fx - fw * margin))
        y1 = max(0, int(fy - fh * margin))
        x2 = min(w, int(fx + fw + fw * margin))
        y2 = min(h, int(fy + fh + fh * margin))

        face_crop = img_np[y1:y2, x1:x2]

        # Normalizar iluminação/contraste (CLAHE) para facilitar comparação
        # com foto da câmara (que tem condições de luz completamente diferentes)
        face_crop = _preprocess_face_for_comparison(face_crop)

        # Melhorar qualidade da face extraída
        face_pil = Image.fromarray(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

        # Redimensionar para tamanho normalizado (mantendo proporção)
        target_size = 400
        ratio = target_size / max(face_pil.size)
        new_size = (int(face_pil.size[0] * ratio), int(face_pil.size[1] * ratio))
        face_pil = face_pil.resize(new_size, Image.LANCZOS)

        # Melhorar nitidez levemente
        enhancer = ImageEnhance.Sharpness(face_pil)
        face_pil = enhancer.enhance(1.3)

        # Melhorar contraste levemente
        enhancer = ImageEnhance.Contrast(face_pil)
        face_pil = enhancer.enhance(1.1)

        # Converter para bytes
        buffer = io.BytesIO()
        face_pil.save(buffer, format="JPEG", quality=95)
        face_bytes = buffer.getvalue()
        face_b64 = base64.b64encode(face_bytes).decode("utf-8")

        # Gerar session_id para o fluxo
        session_id = uuid.uuid4().hex[:16]

        # Guardar face extraída para verificação posterior
        session_dir = os.path.join(DOCUMENTS_DIR, session_id)
        os.makedirs(session_dir, exist_ok=True)
        face_path = os.path.join(session_dir, "document_face.jpg")
        with open(face_path, "wb") as f:
            f.write(face_bytes)

        # Avaliar qualidade da face extraída
        quality = full_quality_assessment(face_bytes)

        # Recolher resultado OCR (correu em paralelo com a extração da face)
        try:
            ocr_data = ocr_future.result(timeout=60)
            logger.debug(f"[DOC OCR] resultado: {ocr_data}")
        except Exception as _e:
            logger.debug(f"[DOC OCR] falhou: {_e}")
            ocr_data = {"success": False}

        return {
            "success": True,
            "session_id": session_id,
            "face_image_b64": face_b64,
            "face_size": {"w": new_size[0], "h": new_size[1]},
            "original_region": {"x": fx, "y": fy, "w": fw, "h": fh},
            "detection_confidence": round(float(best_face.get("confidence", 0)), 4),
            "quality": quality,
            "document_data": ocr_data if ocr_data.get("success") else None,
            "message": "Face extraída com sucesso do documento."
                       + (" Qualidade aprovada." if quality.get("overall_passed") else " ⚠️ Qualidade abaixo do ideal, mas utilizável."),
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "message": f"Erro ao processar documento: {str(e)}"}

    finally:
        executor.shutdown(wait=False)
        _cleanup(doc_path)


def _get_doc_threshold():
    return get_df_runtime_config()["df_document_threshold"]

_ENSEMBLE_DOC_THRESHOLDS = {"ArcFace": DOCUMENT_THRESHOLD}  # fallback estático


def verify_document_vs_live(session_id: str, live_image_bytes: bytes) -> dict:
    """
    Verificação 1:1 - Compara a face extraída do documento com a captura em tempo real.
    Anti-spoofing ativado para a imagem da câmara.

    Args:
        session_id: ID da sessão (da extração do documento)
        live_image_bytes: Bytes da imagem capturada em tempo real

    Returns:
        dict com resultado da verificação
    """
    session_dir = os.path.join(DOCUMENTS_DIR, session_id)
    doc_face_path = os.path.join(session_dir, "document_face.jpg")

    if not os.path.exists(doc_face_path):
        return {
            "success": False,
            "verified": False,
            "message": "Sessão expirada ou inválida. Carregue o documento novamente.",
        }

    t0 = time.perf_counter()
    live_path = _save_temp(live_image_bytes, "live", max_dim=640)
    logger.debug(f"[TIMING DOC] guardar imagem temp: {time.perf_counter()-t0:.3f}s")

    try:
        # Anti-spoofing na imagem ORIGINAL (antes de qualquer pré-processamento).
        # O CLAHE converte a imagem para um estado "processado" que o modelo de
        # anti-spoofing deteta como artificial — por isso, correr anti-spoofing
        # primeiro, como acontece no 1:1 e 1:N.
        from backend.deepface.services.deepface_service import _check_live_anti_spoofing, _get_anti_spoofing_runtime_status
        anti_spoofing_active = False
        anti_spoofing_warning = None

        if _get_anti_spoofing_runtime_status():
            t2 = time.perf_counter()
            spoof_check = _check_live_anti_spoofing(live_path)
            logger.debug(f"[TIMING DOC] anti-spoofing check: {time.perf_counter()-t2:.3f}s")
            if not spoof_check["ok"]:
                if spoof_check.get("spoof_detected"):
                    return {
                        "success": False,
                        "verified": False,
                        "spoof_detected": True,
                        "message": spoof_check["message"],
                    }
                return {
                    "success": False,
                    "verified": False,
                    "message": spoof_check.get("message", "Erro na validação de prova de vida."),
                }
            anti_spoofing_active = spoof_check.get("anti_spoofing_available", True)
            anti_spoofing_warning = spoof_check.get("warning")
        else:
            anti_spoofing_warning = "Anti-spoofing indisponível no servidor (dependência 'torch' não instalada)."

        # Pré-processar a imagem ao vivo com CLAHE (depois do anti-spoofing) para
        # normalizar iluminação igual ao que foi feito na foto do documento.
        try:
            t1 = time.perf_counter()
            _live_bgr = cv2.imread(live_path)
            if _live_bgr is not None:
                _live_proc = _preprocess_face_for_comparison(_live_bgr)
                cv2.imwrite(live_path, _live_proc)
            logger.debug(f"[TIMING DOC] CLAHE preprocessing: {time.perf_counter()-t1:.3f}s")
        except Exception:
            pass  # Se falhar, continua com a imagem original

        # Verificação com ArcFace (99.83% LFW — melhor precisão disponível)
        t3 = time.perf_counter()
        result = DeepFace.verify(
            img1_path=doc_face_path,
            img2_path=live_path,
            model_name="ArcFace",
            detector_backend=DETECTOR_BACKEND,
            distance_metric=DISTANCE_METRIC,
            anti_spoofing=False,
            threshold=_get_doc_threshold(),
        )
        logger.debug(f"[TIMING DOC] DeepFace.verify: {time.perf_counter()-t3:.3f}s")
        logger.debug(f"[TIMING DOC] TOTAL: {time.perf_counter()-t0:.3f}s")
        verified = bool(result["verified"])
        distance = round(float(result["distance"]), 6)
        threshold = round(float(result["threshold"]), 6)

        return {
            "success": True,
            "verified": verified,
            "distance": distance,
            "threshold": threshold,
            "anti_spoofing": anti_spoofing_active,
            "anti_spoofing_warning": anti_spoofing_warning,
            "message": "✅ Identidade verificada com sucesso! A face corresponde ao documento."
                       if verified
                       else "❌ As faces NÃO correspondem. A face da câmara não coincide com o documento.",
        }

    except ValueError as e:
        error_msg = str(e)
        if "Spoof" in error_msg or "spoof" in error_msg:
            return {
                "success": False,
                "verified": False,
                "spoof_detected": True,
                "message": "🚨 Spoofing detetado! Certifique-se de que está a usar a câmara em tempo real com o seu rosto real.",
            }
        elif "Face" in error_msg:
            return {
                "success": False,
                "verified": False,
                "message": "Nenhuma face detetada na câmara. Certifique-se de que o seu rosto está visível.",
            }
        return {"success": False, "verified": False, "message": f"Erro: {error_msg}"}

    except Exception as e:
        return {"success": False, "verified": False, "message": f"Erro inesperado: {str(e)}"}

    finally:
        _cleanup(live_path)
        # Limpar sessão após verificação — a face do documento já não é necessária
        shutil.rmtree(session_dir, ignore_errors=True)


# === OCR: Extração de Dados Textuais do Documento ===

_ocr_reader = None


def _get_ocr_reader():
    """Inicializa o leitor EasyOCR (lazy-loaded, partilhado entre pedidos)."""
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(['pt', 'en'], gpu=False, verbose=False)
    return _ocr_reader


def _parse_document_ocr(ocr_results: list) -> dict:
    """Extrai data de nascimento e calcula a idade a partir dos resultados do OCR."""
    import re
    from datetime import date as Date

    # Juntar todo o texto detetado numa string única para regex
    all_text = " ".join(
        text.strip().upper()
        for _, text, conf in ocr_results
        if conf > 0.25 and text.strip()
    )

    MONTHS_MAP = {
        'JAN': 1, 'FEV': 2, 'FEB': 2, 'MAR': 3, 'ABR': 4, 'APR': 4,
        'MAI': 5, 'MAY': 5, 'JUN': 6, 'JUL': 7, 'AGO': 8, 'AUG': 8,
        'SET': 9, 'SEP': 9, 'OUT': 10, 'OCT': 10, 'NOV': 11, 'DEZ': 12, 'DEC': 12,
    }
    dob = None
    dob_str = None

    # Padrão 1: DD MM YYYY / DD-MM-YYYY / DD.MM.YYYY / DD/MM/YYYY
    for m in re.finditer(r'\b(\d{1,2})[.\-\s/](\d{1,2})[.\-\s/](\d{4})\b', all_text):
        d, mo, y = int(m.group(1)), int(m.group(2)), int(m.group(3))
        if 1 <= d <= 31 and 1 <= mo <= 12 and 1900 <= y <= 2015:
            try:
                dob = Date(y, mo, d)
                dob_str = f"{d:02d}-{mo:02d}-{y}"
                break
            except ValueError:
                continue

    # Padrão 2: DD MON YYYY  (ex. "10 AGO 2005")
    if not dob:
        for m in re.finditer(r'\b(\d{1,2})\s+([A-Z]{3})\.?\s+(\d{4})\b', all_text):
            d, mon, y = int(m.group(1)), m.group(2)[:3], int(m.group(3))
            if mon in MONTHS_MAP and 1 <= d <= 31 and 1900 <= y <= 2015:
                try:
                    dob = Date(y, MONTHS_MAP[mon], d)
                    dob_str = f"{d:02d}-{MONTHS_MAP[mon]:02d}-{y}"
                    break
                except ValueError:
                    continue

    # Padrão 3: MRZ ICAO 9303 — YYMMDD em linha sem '<<'
    if not dob:
        for _, text, conf in ocr_results:
            if conf < 0.25:
                continue
            clean = text.strip().upper().replace(' ', '')
            if len(clean) >= 28 and re.match(r'^[A-Z0-9<]+$', clean) and '<' in clean and '<<' not in clean:
                for m in re.finditer(r'(\d{2})(\d{2})(\d{2})', clean):
                    yy, mm, dd = int(m.group(1)), int(m.group(2)), int(m.group(3))
                    if 1 <= mm <= 12 and 1 <= dd <= 31:
                        yr = (2000 + yy) if yy <= 25 else (1900 + yy)
                        if 1900 <= yr <= 2015:
                            try:
                                dob = Date(yr, mm, dd)
                                dob_str = f"{dob.day:02d}-{dob.month:02d}-{dob.year}"
                                break
                            except ValueError:
                                pass
                if dob:
                    break

    def _calc_age(d):
        if not d:
            return None
        today = Date.today()
        a = today.year - d.year - ((today.month, today.day) < (d.month, d.day))
        return a if 0 <= a <= 120 else None

    age = _calc_age(dob)
    if age is None:
        dob = None
        dob_str = None

    return {"date_of_birth": dob_str, "age": age}


def extract_document_ocr_data(image_bytes: bytes) -> dict:
    """
    Extrai a data de nascimento e a idade do documento via OCR.
    Tenta 4 orientações (0°, 90°CW, 90°CCW, 180°) e para assim que encontrar uma data.
    """

    try:
        reader = _get_ocr_reader()
    except Exception as e:
        return {"success": False, "error": f"OCR indisponível: {e}"}

    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return {"success": False, "error": "Imagem inválida"}

        # Normalizar para 800px
        h, w = img.shape[:2]
        scale = 800 / max(h, w)
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)

        # CLAHE para melhorar contraste
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        img_proc = cv2.cvtColor(
            cv2.merge([clahe.apply(l_ch), a_ch, b_ch]), cv2.COLOR_LAB2BGR
        )

        # Tentar as 4 orientações; parar assim que encontrar uma data de nascimento
        orientations = [
            (img_proc,                                                "0°"),
            (cv2.rotate(img_proc, cv2.ROTATE_90_CLOCKWISE),        "90°CW"),
            (cv2.rotate(img_proc, cv2.ROTATE_90_COUNTERCLOCKWISE), "90°CCW"),
            (cv2.rotate(img_proc, cv2.ROTATE_180),                 "180°"),
        ]

        for rot_img, angle in orientations:
            results = reader.readtext(rot_img)
            parsed = _parse_document_ocr(results)
            logger.debug(f"[OCR] {angle}: age={parsed.get('age')}  dob={parsed.get('date_of_birth')!r}")
            if parsed.get("age") is not None:
                parsed["success"] = True
                return parsed

        # Nenhuma orientação encontrou data
        return {"success": True, "date_of_birth": None, "age": None}

    except Exception as e:
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


def cleanup_session(session_id: str) -> dict:
    """Remove dados temporários de uma sessão."""
    session_dir = os.path.join(DOCUMENTS_DIR, session_id)
    if os.path.exists(session_dir):
        shutil.rmtree(session_dir, ignore_errors=True)
        return {"success": True, "message": "Sessão limpa."}
    return {"success": False, "message": "Sessão não encontrada."}


def cleanup_old_sessions(max_age_minutes: int = 30) -> int:
    """Remove sessões de documento mais antigas que max_age_minutes. Retorna o número removido."""
    import time
    if not os.path.exists(DOCUMENTS_DIR):
        return 0
    removed = 0
    cutoff = time.time() - max_age_minutes * 60
    for name in os.listdir(DOCUMENTS_DIR):
        session_dir = os.path.join(DOCUMENTS_DIR, name)
        if os.path.isdir(session_dir):
            try:
                if os.path.getmtime(session_dir) < cutoff:
                    shutil.rmtree(session_dir, ignore_errors=True)
                    removed += 1
            except OSError:
                pass
    return removed

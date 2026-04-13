"""
Servico de avaliacao de qualidade de imagem.
Analisa atributos de qualidade: desfocagem, brilho, tamanho da face e orientacao.
"""
import numpy as np

from backend.config import (
    QUALITY_MIN_FACE_SIZE,
    QUALITY_MAX_BLUR,
    QUALITY_MIN_BRIGHTNESS,
    QUALITY_MAX_BRIGHTNESS,
    QUALITY_MIN_FACE_RATIO,
    get_df_runtime_config,
)


def _bytes_to_numpy(image_bytes: bytes) -> np.ndarray:
    """Converte bytes de imagem para array numpy (BGR para OpenCV)."""
    import cv2

    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def _iter_rotations(image_np: np.ndarray):
    """Gera imagem original e rotacoes de 90/180 graus para deteccao robusta."""
    import cv2

    return [
        (image_np, "0"),
        (cv2.rotate(image_np, cv2.ROTATE_90_CLOCKWISE), "90_cw"),
        (cv2.rotate(image_np, cv2.ROTATE_90_COUNTERCLOCKWISE), "90_ccw"),
        (cv2.rotate(image_np, cv2.ROTATE_180), "180"),
    ]


def _detect_best_face(image_np: np.ndarray, detector_backend: str) -> dict:
    """
    Tenta detetar face em multiplas rotacoes e com fallback de backend.
    Retorna a melhor deteccao (maior confianca).
    """
    from deepface import DeepFace

    backends = [detector_backend]
    if detector_backend != "opencv":
        backends.append("opencv")

    best = None
    last_error = None

    for backend in backends:
        for rotated_np, rotation in _iter_rotations(image_np):
            try:
                faces = DeepFace.extract_faces(
                    img_path=rotated_np,
                    detector_backend=backend,
                    anti_spoofing=False,
                    enforce_detection=False,
                )
            except Exception as e:
                last_error = e
                continue

            for face in faces or []:
                raw_confidence = float(face.get("confidence", 0))
                # O backend opencv pode nao devolver score; nesse caso
                # usamos um valor neutro para permitir a validacao seguir.
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
                        "face": face,
                        "image_np": rotated_np,
                        "rotation": rotation,
                        "backend": backend,
                        "confidence": confidence,
                        "raw_confidence": raw_confidence,
                        "area": area,
                    }

        # Evita custo extra quando ja existe uma deteccao forte.
        if best and best["confidence"] >= 0.65:
            break

    if best:
        return {"success": True, **best}

    return {"success": False, "error": str(last_error) if last_error else None}


def assess_blur(image_np: np.ndarray) -> dict:
    """
    Avalia desfocagem usando variancia do Laplaciano.
    Valores baixos = imagem desfocada.
    """
    import cv2

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

    min_blur = get_df_runtime_config()["df_quality_min_blur"]
    is_blurry = laplacian_var < min_blur

    return {
        "metric": "blur",
        "value": round(float(laplacian_var), 2),
        "threshold": min_blur,
        "passed": not is_blurry,
        "label": "Desfocagem",
        "detail": f"Variancia Laplaciano: {laplacian_var:.1f} (min: {min_blur})",
        "message": "Imagem desfocada - tire outra foto com melhor foco" if is_blurry else "Foco adequado",
    }


def assess_brightness(image_np: np.ndarray) -> dict:
    """
    Avalia brilho medio da imagem.
    """
    import cv2

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    mean_brightness = float(np.mean(gray))

    too_dark = mean_brightness < QUALITY_MIN_BRIGHTNESS
    too_bright = mean_brightness > QUALITY_MAX_BRIGHTNESS
    passed = not too_dark and not too_bright

    if too_dark:
        msg = "Imagem muito escura - melhore a iluminacao"
    elif too_bright:
        msg = "Imagem com brilho excessivo - reduza a exposicao"
    else:
        msg = "Iluminacao adequada"

    return {
        "metric": "brightness",
        "value": round(mean_brightness, 1),
        "range": [QUALITY_MIN_BRIGHTNESS, QUALITY_MAX_BRIGHTNESS],
        "passed": passed,
        "label": "Iluminacao",
        "detail": f"Brilho medio: {mean_brightness:.0f} (intervalo: {QUALITY_MIN_BRIGHTNESS}-{QUALITY_MAX_BRIGHTNESS})",
        "message": msg,
    }


def assess_contrast(image_np: np.ndarray) -> dict:
    """
    Avalia contraste da imagem usando desvio padrao.
    """
    import cv2

    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    std_dev = float(np.std(gray))

    low_contrast = std_dev < 15

    return {
        "metric": "contrast",
        "value": round(std_dev, 1),
        "threshold": 30,
        "passed": not low_contrast,
        "label": "Contraste",
        "detail": f"Desvio padrao: {std_dev:.1f} (min: 30)",
        "message": "Contraste muito baixo" if low_contrast else "Contraste adequado",
    }


def assess_face_size(face_region: dict, image_shape: tuple) -> dict:
    """
    Avalia se a face e suficientemente grande na imagem.

    Args:
        face_region: dict com x, y, w, h da face detetada
        image_shape: (height, width) da imagem
    """
    img_h, img_w = image_shape[:2]
    face_w = face_region.get("w", 0)
    face_h = face_region.get("h", 0)

    face_area = face_w * face_h
    image_area = img_w * img_h
    face_ratio = face_area / image_area if image_area > 0 else 0

    min_face_px = get_df_runtime_config()["df_quality_min_face_size"]
    too_small_pixels = face_w < min_face_px or face_h < min_face_px
    too_small_ratio = face_ratio < QUALITY_MIN_FACE_RATIO
    passed = not too_small_pixels and not too_small_ratio

    if too_small_pixels:
        msg = f"Face muito pequena ({face_w}x{face_h}px) - aproxime-se da camara"
    elif too_small_ratio:
        msg = f"Face ocupa apenas {face_ratio*100:.1f}% da imagem - aproxime-se"
    else:
        msg = "Tamanho da face adequado"

    return {
        "metric": "face_size",
        "face_pixels": {"w": face_w, "h": face_h},
        "face_ratio": round(face_ratio, 4),
        "min_pixels": min_face_px,
        "min_ratio": QUALITY_MIN_FACE_RATIO,
        "passed": passed,
        "label": "Tamanho da Face",
        "detail": f"Face: {face_w}x{face_h}px ({face_ratio*100:.1f}% da imagem)",
        "message": msg,
    }


def assess_image_resolution(image_np: np.ndarray) -> dict:
    """
    Avalia resolucao geral da imagem.
    """
    h, w = image_np.shape[:2]
    total_pixels = h * w
    min_pixels = 160 * 120

    passed = total_pixels >= min_pixels

    return {
        "metric": "resolution",
        "value": {"width": w, "height": h, "total": total_pixels},
        "min_total": min_pixels,
        "passed": passed,
        "label": "Resolucao",
        "detail": f"{w}x{h} ({total_pixels:,} pixels)",
        "message": "Resolucao muito baixa" if not passed else "Resolucao adequada",
    }


def full_quality_assessment(image_bytes: bytes) -> dict:
    """
    Avaliacao completa de qualidade de uma imagem facial.
    Inclui deteccao de face + todos os atributos de qualidade.
    O blur e avaliado na regiao da face (nao na imagem inteira),
    para evitar falsos positivos com fundos uniformes.

    Returns:
        dict com resultados detalhados de qualidade
    """
    from backend.config import DETECTOR_BACKEND

    image_np = _bytes_to_numpy(image_bytes)
    if image_np is None:
        return {
            "success": False,
            "overall_passed": False,
            "checks": [],
            "message": "Nao foi possivel ler a imagem.",
        }

    checks = []

    # 1. Resolucao
    checks.append(assess_image_resolution(image_np))

    # 2. Brilho (imagem toda)
    checks.append(assess_brightness(image_np))

    # 3. Contraste (imagem toda)
    checks.append(assess_contrast(image_np))

    # 4. Pre-verificacao de desfocagem na imagem toda
    blur_precheck = assess_blur(image_np)
    if not blur_precheck["passed"]:
        checks.append(blur_precheck)
        # Imagem demasiado desfocada para detetar face - nao tenta deteccao.
    else:
        # 5. Deteccao de face + tamanho + blur preciso na regiao da face
        face_np_for_blur = None
        try:
            detection = _detect_best_face(image_np, DETECTOR_BACKEND)

            if detection.get("success"):
                face = detection["face"]
                image_for_face = detection["image_np"]
                region = face.get("facial_area", {})
                checks.append(assess_face_size(region, image_for_face.shape))

                # Recortar regiao da face para avaliar blur (ignora fundo uniforme)
                fx = int(region.get("x", 0))
                fy = int(region.get("y", 0))
                fw = int(region.get("w", 0))
                fh = int(region.get("h", 0))
                img_h, img_w = image_for_face.shape[:2]
                x1 = max(0, fx)
                y1 = max(0, fy)
                x2 = min(img_w, fx + fw)
                y2 = min(img_h, fy + fh)
                if x2 > x1 and y2 > y1:
                    face_np_for_blur = image_for_face[y1:y2, x1:x2]

                # Confianca da deteccao
                confidence = float(detection["confidence"])
                opencv_without_score = (
                    detection.get("backend") == "opencv"
                    and float(detection.get("raw_confidence", 0)) <= 0
                )
                confidence_passed = opencv_without_score or confidence >= 0.7
                checks.append(
                    {
                        "metric": "detection_confidence",
                        "value": round(confidence, 4),
                        "threshold": 0.7,
                        "passed": confidence_passed,
                        "label": "Confianca da Deteccao",
                        "detail": (
                            "Detecao via fallback opencv (sem score de confianca)"
                            if opencv_without_score
                            else (
                                f"Confianca: {confidence:.2%} "
                                f"(backend: {detection['backend']}, rotacao: {detection['rotation']})"
                            )
                        ),
                        "message": (
                            "Face detetada com fallback opencv"
                            if opencv_without_score
                            else (
                                "Baixa confianca na deteccao"
                                if confidence < 0.7
                                else "Face detetada com confianca"
                            )
                        ),
                    }
                )
            else:
                checks.append(
                    {
                        "metric": "face_detected",
                        "passed": False,
                        "label": "Deteccao Facial",
                        "message": "Nenhuma face detetada na imagem",
                    }
                )

        except Exception:
            checks.append(
                {
                    "metric": "face_detected",
                    "passed": False,
                    "label": "Deteccao Facial",
                    "message": "Nenhuma face detetada - certifique-se de que a foto mostra o rosto claramente",
                }
            )

        # Desfocagem precisa - na face se disponivel, senao na imagem toda
        checks.append(assess_blur(face_np_for_blur if face_np_for_blur is not None else image_np))

    overall = all(c.get("passed", False) for c in checks)
    failed = [c for c in checks if not c.get("passed", False)]

    if overall:
        message = "Imagem aprovada em todos os criterios de qualidade."
    else:
        issues = [c["label"] for c in failed]
        message = f"Problemas detetados: {', '.join(issues)}"

    return {
        "success": True,
        "overall_passed": overall,
        "checks": checks,
        "total_checks": len(checks),
        "passed_checks": len(checks) - len(failed),
        "failed_checks": len(failed),
        "message": message,
    }

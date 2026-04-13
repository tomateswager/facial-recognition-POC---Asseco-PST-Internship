"""
anti_spoofing_service.py - Deteção de Liveness / Anti-Spoofing local
Analisa textura, frequência, cor e qualidade AWS para detetar spoofing.
"""
import cv2
import numpy as np
import logging
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class SpoofingResult:
    is_live: bool
    overall_score: float
    checks: dict = field(default_factory=dict)
    message: str = ""

    def to_dict(self) -> dict:
        return {"is_live": self.is_live, "overall_score": round(self.overall_score, 2), "checks": self.checks, "message": self.message}


class AntiSpoofing:
    def __init__(self, threshold: float = 50.0):
        self.threshold = threshold
        self.weights = {"texture": 0.35, "frequency": 0.20, "color": 0.15, "aws_quality": 0.30}

    def _bytes_to_cv2(self, image_bytes: bytes) -> np.ndarray:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("Não foi possível descodificar a imagem")
        return img

    def _bytes_to_gray(self, image_bytes: bytes) -> np.ndarray:
        return cv2.cvtColor(self._bytes_to_cv2(image_bytes), cv2.COLOR_BGR2GRAY)

    def check_texture(self, image_bytes: bytes) -> dict:
        try:
            gray = self._bytes_to_gray(image_bytes)
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()
            if variance > 300: score = 100.0
            elif variance > 100: score = 60 + (variance - 100) * (40 / 200)
            elif variance > 50: score = 30 + (variance - 50) * (30 / 50)
            else: score = max(0, variance * (30 / 50))
            return {"score": min(100, score), "variance": round(variance, 2), "passed": score >= 50, "detail": f"Variância Laplaciana: {variance:.1f}"}
        except Exception as e:
            return {"score": 50, "variance": 0, "passed": True, "detail": f"Erro: {e}"}

    def check_frequency(self, image_bytes: bytes) -> dict:
        try:
            gray = cv2.resize(self._bytes_to_gray(image_bytes), (256, 256))
            f_shift = np.fft.fftshift(np.fft.fft2(gray))
            magnitude = np.log1p(np.abs(f_shift))
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            y, x = np.ogrid[:h, :w]
            low_mask = (y - cy)**2 + (x - cx)**2 <= 30**2
            ratio = magnitude[~low_mask].mean() / (magnitude[low_mask].mean() + 1e-6)
            if ratio > 0.35: score = 100.0
            elif ratio > 0.20: score = 50 + (ratio - 0.20) * (50 / 0.15)
            else: score = max(0, ratio * (50 / 0.20))
            return {"score": min(100, score), "frequency_ratio": round(ratio, 4), "passed": score >= 40, "detail": f"Rácio freq: {ratio:.4f}"}
        except Exception as e:
            return {"score": 50, "frequency_ratio": 0, "passed": True, "detail": f"Erro: {e}"}

    def check_color(self, image_bytes: bytes) -> dict:
        try:
            img = self._bytes_to_cv2(image_bytes)
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
            color_diversity = (hsv[:,:,1].std() + ycrcb[:,:,1].astype(float).std() + ycrcb[:,:,2].astype(float).std()) / 3
            if color_diversity > 30: score = 90.0
            elif color_diversity > 20: score = 50 + (color_diversity - 20) * (40 / 10)
            elif color_diversity > 10: score = 20 + (color_diversity - 10) * (30 / 10)
            else: score = max(0, color_diversity * 2)
            return {"score": min(100, score), "color_diversity": round(color_diversity, 2), "passed": score >= 40, "detail": f"Diversidade cromática: {color_diversity:.1f}"}
        except Exception as e:
            return {"score": 50, "color_diversity": 0, "passed": True, "detail": f"Erro: {e}"}

    def check_aws_quality(self, aws_face_data: dict) -> dict:
        try:
            if not aws_face_data.get("faces"):
                return {"score": 0, "passed": False, "detail": "Nenhuma face detetada"}
            face = aws_face_data["faces"][0]
            quality = face.get("quality", {})
            pose = face.get("pose", {})
            sharpness = quality.get("sharpness", 0)
            brightness = quality.get("brightness", 0)
            confidence = face.get("confidence", 0)
            eyes_open = face.get("eyes_open", None)
            eyes_conf = face.get("eyes_open_confidence", 0)
            sunglasses = face.get("sunglasses", False)
            score = (confidence / 100) * 20
            score += min(sharpness / 100, 1.0) * 25
            score += 15 if 40 <= brightness <= 80 else (10 if 20 <= brightness <= 90 else 5)
            score += (25 if eyes_conf > 80 else 15) if eyes_open else 5
            yaw, pitch = abs(pose.get("yaw", 0)), abs(pose.get("pitch", 0))
            score += 15 if yaw < 15 and pitch < 15 else (10 if yaw < 30 and pitch < 25 else 5)
            if sunglasses: score *= 0.7
            return {"score": min(100, score), "passed": score >= 50, "sharpness": sharpness, "brightness": brightness, "detail": f"Conf:{confidence:.0f}% Nitidez:{sharpness:.0f} Brilho:{brightness:.0f}"}
        except Exception as e:
            return {"score": 50, "passed": True, "detail": f"Erro: {e}"}

    def analyze_single_frame(self, image_bytes: bytes, aws_face_data: Optional[dict] = None) -> SpoofingResult:
        checks = {
            "texture": self.check_texture(image_bytes),
            "frequency": self.check_frequency(image_bytes),
            "color": self.check_color(image_bytes),
            "aws_quality": self.check_aws_quality(aws_face_data) if aws_face_data else {"score": 50, "passed": True, "detail": "N/A"},
        }
        overall = sum(checks[k]["score"] * w for k, w in self.weights.items())
        is_live = overall >= self.threshold
        return SpoofingResult(is_live=is_live, overall_score=overall, checks=checks,
                              message="Face REAL detetada" if is_live else "Possível SPOOF detetado")

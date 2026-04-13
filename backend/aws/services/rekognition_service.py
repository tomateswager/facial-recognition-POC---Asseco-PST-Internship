"""
rekognition_service.py - Wrapper para AWS Rekognition
Encapsula operações: CompareFaces, IndexFaces, SearchFacesByImage, DetectFaces, gestão de coleções.
"""
import boto3
import time
import logging
import os
import io
from typing import Optional
from botocore.exceptions import ClientError, NoCredentialsError

logger = logging.getLogger(__name__)


_AWS_MAX_BYTES = 4_900_000  # AWS limite: 5MB; usamos 4.9MB para margem
_AWS_MAX_DIM   = 2048       # Redimensionar se maior que isto


def _to_jpeg(image_bytes: bytes) -> bytes:
    """Converte para JPEG e redimensiona se necessário (AWS: máx 5MB, JPEG/PNG)."""
    from PIL import Image
    try:
        img = Image.open(io.BytesIO(image_bytes))
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")

        # Redimensionar se demasiado grande
        w, h = img.size
        if max(w, h) > _AWS_MAX_DIM:
            ratio = _AWS_MAX_DIM / max(w, h)
            img = img.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)

        # Converter para JPEG com qualidade progressivamente menor até caber
        for quality in (92, 80, 65, 50):
            buf = io.BytesIO()
            img.save(buf, format="JPEG", quality=quality)
            result = buf.getvalue()
            if len(result) <= _AWS_MAX_BYTES:
                return result

        # Último recurso: reduzir dimensão a metade
        img = img.resize((img.width // 2, img.height // 2), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=50)
        return buf.getvalue()
    except Exception as e:
        logger.warning(f"[Rekognition] Não foi possível converter imagem: {e}")
        return image_bytes


class RekognitionClient:
    def __init__(self):
        access_key = os.getenv("AWS_ACCESS_KEY_ID", "")
        secret_key = os.getenv("AWS_SECRET_ACCESS_KEY", "")
        region = os.getenv("AWS_REGION", "eu-west-1")
        if not access_key or not secret_key:
            raise ValueError("Credenciais AWS não configuradas no .env")
        self.client = boto3.client(
            "rekognition",
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            region_name=region,
        )
        self.region = region

    def _timed_call(self, operation: str, func, **kwargs):
        start = time.perf_counter()
        response = func(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        return response, latency_ms

    def test_connection(self) -> dict:
        try:
            response, latency = self._timed_call("TestConnection", self.client.list_collections, MaxResults=1)
            return {"success": True, "latency_ms": round(latency, 2), "region": self.region}
        except NoCredentialsError:
            return {"success": False, "message": "Credenciais AWS não configuradas"}
        except ClientError as e:
            return {"success": False, "message": f"Erro AWS: {e.response['Error']['Message']}"}

    def create_collection(self, collection_id: str) -> dict:
        try:
            response, latency = self._timed_call("CreateCollection", self.client.create_collection, CollectionId=collection_id)
            return {"success": True, "collection_id": collection_id, "latency_ms": round(latency, 2)}
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceAlreadyExistsException":
                return {"success": True, "collection_id": collection_id, "message": "Coleção já existe"}
            raise

    def list_faces_in_collection(self, collection_id: str) -> list:
        all_faces = []
        next_token = None
        while True:
            kwargs = {"CollectionId": collection_id, "MaxResults": 100}
            if next_token:
                kwargs["NextToken"] = next_token
            response, _ = self._timed_call("ListFaces", self.client.list_faces, **kwargs)
            all_faces.extend(response.get("Faces", []))
            next_token = response.get("NextToken")
            if not next_token:
                break
        return all_faces

    def detect_faces(self, image_bytes: bytes) -> dict:
        image_bytes = _to_jpeg(image_bytes)
        response, latency = self._timed_call("DetectFaces", self.client.detect_faces,
            Image={"Bytes": image_bytes}, Attributes=["ALL"])
        faces = []
        for face in response.get("FaceDetails", []):
            faces.append({
                "confidence": face.get("Confidence", 0),
                "bounding_box": face.get("BoundingBox", {}),
                "pose": {"yaw": face.get("Pose", {}).get("Yaw", 0), "pitch": face.get("Pose", {}).get("Pitch", 0), "roll": face.get("Pose", {}).get("Roll", 0)},
                "quality": {"sharpness": face.get("Quality", {}).get("Sharpness", 0), "brightness": face.get("Quality", {}).get("Brightness", 0)},
                "eyes_open": face.get("EyesOpen", {}).get("Value", None),
                "eyes_open_confidence": face.get("EyesOpen", {}).get("Confidence", 0),
                "mouth_open": face.get("MouthOpen", {}).get("Value", None),
                "mouth_open_confidence": face.get("MouthOpen", {}).get("Confidence", 0),
                "smile": face.get("Smile", {}).get("Value", None),
                "smile_confidence": face.get("Smile", {}).get("Confidence", 0),
                "eyeglasses": face.get("Eyeglasses", {}).get("Value", None),
                "eyeglasses_confidence": face.get("Eyeglasses", {}).get("Confidence", 0),
                "sunglasses": face.get("Sunglasses", {}).get("Value", None),
                "sunglasses_confidence": face.get("Sunglasses", {}).get("Confidence", 0),
                "beard": face.get("Beard", {}).get("Value", None),
                "beard_confidence": face.get("Beard", {}).get("Confidence", 0),
                "mustache": face.get("Mustache", {}).get("Value", None),
                "mustache_confidence": face.get("Mustache", {}).get("Confidence", 0),
                "face_occluded": face.get("FaceOccluded", {}).get("Value", None),
                "face_occluded_confidence": face.get("FaceOccluded", {}).get("Confidence", 0),
                "age_range": {"low": face.get("AgeRange", {}).get("Low", 0), "high": face.get("AgeRange", {}).get("High", 0)},
                "gender": face.get("Gender", {}).get("Value", None),
                "gender_confidence": face.get("Gender", {}).get("Confidence", 0),
                "emotions": [{"type": e.get("Type"), "confidence": e.get("Confidence", 0)} for e in sorted(face.get("Emotions", []), key=lambda x: x.get("Confidence", 0), reverse=True)],
                "landmarks": face.get("Landmarks", []),
            })
        return {"face_count": len(faces), "faces": faces, "latency_ms": round(latency, 2)}

    def compare_faces(self, source_bytes: bytes, target_bytes: bytes, similarity_threshold: float = 80.0) -> dict:
        source_bytes = _to_jpeg(source_bytes)
        target_bytes = _to_jpeg(target_bytes)
        # SimilarityThreshold=0 para obter SEMPRE o score real de todas as faces.
        # Aplicamos o threshold nós próprios para não perder o valor de similaridade.
        response, latency = self._timed_call("CompareFaces", self.client.compare_faces,
            SourceImage={"Bytes": source_bytes}, TargetImage={"Bytes": target_bytes},
            SimilarityThreshold=0)
        all_matches = [{"similarity": m.get("Similarity", 0), "face": {"confidence": m.get("Face", {}).get("Confidence", 0), "bounding_box": m.get("Face", {}).get("BoundingBox", {}), "quality": m.get("Face", {}).get("Quality", {}), "pose": m.get("Face", {}).get("Pose", {})}} for m in response.get("FaceMatches", [])]
        best_similarity = all_matches[0]["similarity"] if all_matches else 0.0
        is_match = best_similarity >= similarity_threshold
        matches = [m for m in all_matches if m["similarity"] >= similarity_threshold]
        return {
            "is_match": is_match,
            "matches": matches,
            "best_similarity": best_similarity,
            "unmatched_faces": len([m for m in all_matches if m["similarity"] < similarity_threshold]),
            "threshold_used": similarity_threshold,
            "latency_ms": round(latency, 2),
        }

    def index_face(self, collection_id: str, image_bytes: bytes, external_id: str) -> dict:
        image_bytes = _to_jpeg(image_bytes)
        response, latency = self._timed_call("IndexFaces", self.client.index_faces,
            CollectionId=collection_id, Image={"Bytes": image_bytes},
            ExternalImageId=external_id, MaxFaces=1, QualityFilter="AUTO", DetectionAttributes=["ALL"])
        indexed = [{"face_id": r.get("Face", {}).get("FaceId", ""), "external_id": r.get("Face", {}).get("ExternalImageId", ""), "confidence": r.get("Face", {}).get("Confidence", 0)} for r in response.get("FaceRecords", [])]
        unindexed = [{"reason": r.get("Reasons", [])} for r in response.get("UnindexedFaces", [])]
        return {"success": len(indexed) > 0, "indexed_faces": indexed, "unindexed_faces": unindexed, "latency_ms": round(latency, 2)}

    def search_faces_by_image(self, collection_id: str, image_bytes: bytes, max_faces: int = 5, threshold: float = 80.0) -> dict:
        image_bytes = _to_jpeg(image_bytes)
        response, latency = self._timed_call("SearchFacesByImage", self.client.search_faces_by_image,
            CollectionId=collection_id, Image={"Bytes": image_bytes},
            MaxFaces=max_faces, FaceMatchThreshold=threshold)
        matches = [{"similarity": m.get("Similarity", 0), "face_id": m.get("Face", {}).get("FaceId", ""), "external_id": m.get("Face", {}).get("ExternalImageId", ""), "confidence": m.get("Face", {}).get("Confidence", 0)} for m in response.get("FaceMatches", [])]
        return {
            "match_count": len(matches),
            "matches": matches,
            "searched_face_confidence": response.get("SearchedFaceConfidence", 0),
            "threshold_used": threshold,
            "latency_ms": round(latency, 2),
        }

    def create_face_liveness_session(self) -> dict:
        response, latency = self._timed_call(
            "CreateFaceLivenessSession",
            self.client.create_face_liveness_session,
        )
        return {"session_id": response.get("SessionId", ""), "latency_ms": round(latency, 2)}

    def delete_face(self, collection_id: str, *face_ids: str) -> dict:
        response, latency = self._timed_call("DeleteFaces", self.client.delete_faces,
            CollectionId=collection_id, FaceIds=list(face_ids))
        return {"deleted": response.get("DeletedFaces", []), "latency_ms": round(latency, 2)}

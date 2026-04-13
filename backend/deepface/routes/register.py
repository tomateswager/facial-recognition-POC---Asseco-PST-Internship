"""
Rotas de Registo Biométrico
Gestão de rostos na base de dados (registo, listagem, remoção).
"""
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from backend.config import MAX_IMAGE_BYTES
from backend.deepface.services.deepface_service import (
    register_face,
    get_registered_people,
    delete_person,
    get_system_info,
)

router = APIRouter(prefix="/api/register", tags=["Registo Biométrico"])


@router.post("/")
async def register_new_face(
    image: UploadFile = File(..., description="Imagem facial para registo"),
    name: str = Form(..., description="Nome da pessoa"),
    mode: str = Form("file", description="Modo: 'file' ou 'camera'"),
):
    """
    Registo biométrico - Adiciona um rosto à base de dados.
    mode=camera ativa anti-spoofing.
    """
    if not name or not name.strip():
        raise HTTPException(status_code=400, detail="O nome é obrigatório.")

    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="A imagem é obrigatória.")
    if len(image_bytes) > MAX_IMAGE_BYTES:
        raise HTTPException(status_code=413, detail="Imagem demasiado grande (máximo 10 MB).")

    use_anti_spoofing = (mode == "camera")
    result = register_face(image_bytes, name.strip(), use_anti_spoofing=use_anti_spoofing)
    return result


@router.get("/people")
async def list_registered_people():
    people = get_registered_people()
    return {
        "success": True,
        "people": people,
        "total_people": len(people),
        "total_images": sum(p["num_images"] for p in people),
    }


@router.delete("/people/{person_name}")
async def remove_person(person_name: str):
    return delete_person(person_name)


@router.get("/system-info")
async def system_info():
    return get_system_info()

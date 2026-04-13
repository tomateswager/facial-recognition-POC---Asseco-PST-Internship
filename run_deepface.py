import os

import uvicorn


def _read_bool_env(var_name: str, default: bool) -> bool:
    value = os.getenv(var_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


if __name__ == "__main__":
    host = os.getenv("DEEPFACE_HOST", "0.0.0.0")
    port = int(os.getenv("DEEPFACE_PORT", "8001"))
    reload = _read_bool_env("DEEPFACE_RELOAD", True)

    uvicorn.run(
        "backend.deepface_webservice:app",
        host=host,
        port=port,
        reload=reload,
    )

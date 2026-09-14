from fastapi import APIRouter

router = APIRouter()


@router.post("/qr/generate")
def generate_qr():
    return {
        "status": "success",
        "message": "QR generation endpoint is ready"
    }
from fastapi import APIRouter
from backend.services.noseprint_service import process_noseprint

router = APIRouter()


@router.post("/identify")
def identify_dog():
    result = process_noseprint()

    return result
from fastapi import APIRouter

router = APIRouter()


@router.post("/register")
def register_dog():
    return {
        "status": "success",
        "message": "Dog registration endpoint is ready"
    }
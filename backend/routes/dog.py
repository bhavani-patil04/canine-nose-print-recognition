from fastapi import APIRouter

router = APIRouter()


@router.get("/dog/{dog_id}")
def get_dog(dog_id: str):
    return {
        "status": "success",
        "dog_id": dog_id,
        "message": "Dog lookup endpoint is ready"
    }
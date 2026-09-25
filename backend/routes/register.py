from fastapi import APIRouter
from pydantic import BaseModel

from backend.services.privacy_service import PrivacyService

router = APIRouter(prefix="/dogs", tags=["Dog Registration"])
privacy_service = PrivacyService()


class DogRegistration(BaseModel):
    dog_id: str
    name: str
    location: str
    vaccination_status: str
    sterilization_status: str
    caregiver_phone: str


@router.post("/register")
def register_dog(dog: DogRegistration):
    caregiver_token = privacy_service.generate_caregiver_token()
    encrypted_phone = privacy_service.encrypt_phone(dog.caregiver_phone)

    dog_record = {
        "dog_id": dog.dog_id,
        "name": dog.name,
        "location": dog.location,
        "vaccination_status": dog.vaccination_status,
        "sterilization_status": dog.sterilization_status,
        "caregiver_token": caregiver_token,
        "caregiver_phone_encrypted": encrypted_phone,
    }

    return {
        "status": "success",
        "dog": {
            "dog_id": dog_record["dog_id"],
            "name": dog_record["name"],
            "location": dog_record["location"],
            "vaccination_status": dog_record["vaccination_status"],
            "sterilization_status": dog_record["sterilization_status"],
            "caregiver_token": dog_record["caregiver_token"],
        },
    }
from fastapi import APIRouter, HTTPException

from backend.models.privacy import CaregiverRegistration, CaregiverTokenResponse
from backend.services.privacy_service import PrivacyService

router = APIRouter(prefix="/privacy", tags=["Privacy & Security"])

privacy_service = PrivacyService()
caregiver_store = {}


@router.post("/register-caregiver", response_model=CaregiverTokenResponse)
def register_caregiver(caregiver: CaregiverRegistration):
    caregiver_token = privacy_service.generate_caregiver_token()
    encrypted_phone = privacy_service.encrypt_phone(caregiver.phone_number)
    phone_hash = privacy_service.create_phone_lookup_hash(caregiver.phone_number)

    caregiver_store[caregiver_token] = {
        "encrypted_phone": encrypted_phone,
        "phone_hash": phone_hash,
    }

    return CaregiverTokenResponse(caregiver_token=caregiver_token)


@router.get("/caregiver/{caregiver_token}")
def get_caregiver(caregiver_token: str):
    caregiver = caregiver_store.get(caregiver_token)

    if not caregiver:
        raise HTTPException(status_code=404, detail="Caregiver not found")

    return {
        "caregiver_token": caregiver_token,
        "status": "protected",
    }

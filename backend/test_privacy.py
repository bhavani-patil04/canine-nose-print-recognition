import os

from backend.services.privacy_service import PrivacyService


def test_token_generation_and_phone_encryption():
    os.environ["PRIVACY_ENCRYPTION_KEY"] = "M6lP2Y0b9j1x4V3Qw9f9l4uA0y4Q3l0A0D4C7g6J4Q0="

    service = PrivacyService()
    token = service.generate_caregiver_token()

    assert token.startswith("CARE_")
    assert service.normalize_phone("+91 (987) 654-3210") == "+919876543210"

    encrypted = service.encrypt_phone("+91 9876543210")
    assert encrypted != "+91 9876543210"
    assert service.decrypt_phone(encrypted) == "+919876543210"

    assert service.create_phone_lookup_hash("+91 9876543210")

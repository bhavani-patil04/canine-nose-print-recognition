import hashlib
import hmac
import os
import secrets

from cryptography.fernet import Fernet
from dotenv import load_dotenv

load_dotenv()


class PrivacyService:
    def __init__(self):
        encryption_key = os.getenv("PRIVACY_ENCRYPTION_KEY")

        if not encryption_key:
            raise RuntimeError("PRIVACY_ENCRYPTION_KEY is not configured")

        self.fernet = Fernet(encryption_key.encode())

    def generate_caregiver_token(self) -> str:
        random_part = secrets.token_hex(16)
        return f"CARE_{random_part}"

    def normalize_phone(self, phone: str) -> str:
        phone = phone.strip()

        if phone.startswith("+"):
            normalized = "".join(char for char in phone[1:] if char.isdigit())
            return "+" + normalized

        return "".join(char for char in phone if char.isdigit())

    def encrypt_phone(self, phone: str) -> str:
        normalized_phone = self.normalize_phone(phone)
        encrypted = self.fernet.encrypt(normalized_phone.encode())
        return encrypted.decode()

    def decrypt_phone(self, encrypted_phone: str) -> str:
        decrypted = self.fernet.decrypt(encrypted_phone.encode())
        return decrypted.decode()

    def create_phone_lookup_hash(self, phone: str) -> str:
        normalized_phone = self.normalize_phone(phone)
        secret = os.getenv("PRIVACY_ENCRYPTION_KEY")

        if not secret:
            raise RuntimeError("PRIVACY_ENCRYPTION_KEY is not configured")

        digest = hmac.new(secret.encode(), normalized_phone.encode(), hashlib.sha256)
        return digest.hexdigest()

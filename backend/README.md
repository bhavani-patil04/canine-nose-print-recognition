# Canine Nose-Print Recognition – Backend

## Overview
This repository contains the backend setup for the Canine Nose-Print Recognition Prototype. The backend is developed using Python, FastAPI, and Uvicorn. It provides the basic API structure required to connect the dog identification, dog registration, QR fallback, nose-print processing, and WhatsApp integration modules.

## Tech Stack
- Python
- FastAPI
- Uvicorn
- Git
- GitHub
- Swagger UI

## Project Structure

canine-nose-print-recognition/
│
├── backend/
│   ├── main.py
│   ├── routes/
│   │   ├── health.py
│   │   ├── identify.py
│   │   ├── register.py
│   │   ├── dog.py
│   │   ├── qr.py
│   │   └── webhook.py
│   ├── services/
│   │   └── noseprint_service.py
│   ├── models/
│   └── utils/
│
├── venv/
├── .env
├── .env.example
├── .gitignore
└── README.md

## API Endpoints

| Method | Endpoint | Purpose |
|--------|----------|---------|
| GET | `/health` | Checks whether the backend is running |
| POST | `/identify` | Handles nose-print identification requests |
| POST | `/dogs/register` | Handles new dog registration with caregiver privacy protection |
| POST | `/privacy/register-caregiver` | Generates a caregiver token and stores the encrypted phone number |
| GET | `/privacy/caregiver/{caregiver_token}` | Checks privacy-protected caregiver information |
| GET | `/dog/{dog_id}` | Retrieves dog information |
| GET | `/qr/generate` | Handles QR generation |
| GET | `/webhook` | WhatsApp webhook verification |
| POST | `/webhook` | Receives WhatsApp webhook events |

## Health Check
The `/health` endpoint is used to verify that the backend is running correctly.

Example response:

{
    "status": "ok",
    "service": "canine-noseprint-backend"
}

## Nose-Print Identification
The `/identify` endpoint is connected to the nose-print processing service. Currently, the service contains placeholder functionality and returns a response confirming that the nose-print processing service is ready. The actual AI processing pipeline will be connected during the integration stage.

## Caretaker Privacy Protection
We implemented the first version of the caretaker privacy layer for the backend.

### What was implemented
- Caretaker phone numbers are treated as sensitive data.
- A random caregiver token such as `CARE_8f31a92c...` is generated for each caregiver.
- The original phone number is normalized and encrypted before storage.
- The application uses the caregiver token instead of the raw phone number in normal responses.
- Dog operational data such as location remains available as normal profile information.

### Files added
- `backend/services/privacy_service.py`
- `backend/models/privacy.py`
- `backend/routes/privacy.py`

### Privacy logic
The `PrivacyService` handles:
- token generation
- phone normalization
- phone encryption using Fernet
- phone decryption
- phone lookup hash creation

### Privacy flow
1. User provides caretaker phone number.
2. The backend generates a random token.
3. The phone number is encrypted and stored securely.
4. The dog profile stores the caregiver token instead of the actual phone number.
5. Normal API responses never expose the phone number.

### Example response
```json
{
  "status": "success",
  "dog": {
    "dog_id": "BLR-6001",
    "name": "Bruno",
    "location": "Indiranagar",
    "vaccination_status": "Verified",
    "sterilization_status": "Completed",
    "caregiver_token": "CARE_8f31a92c..."
  }
}
```

This ensures the dog profile remains usable while the caretaker's personal contact details remain protected.

## Nose-Print Processing Service
The nose-print processing service is located at:

`backend/services/noseprint_service.py`

Currently, it provides a basic placeholder response:

{
    "status": "success",
    "message": "Nose-print processing service is ready"
}

This service will later be connected to the actual video processing, nose detection, image enhancement, feature extraction, and similarity search pipeline.

## Environment Configuration
Environment configuration has been prepared using `.env` and `.env.example`.

The current configuration structure contains placeholders for:

WHATSAPP_ACCESS_TOKEN=
WHATSAPP_PHONE_NUMBER_ID=
WHATSAPP_VERIFY_TOKEN=

SUPABASE_URL=
SUPABASE_KEY=

DATABASE_URL=

The actual credentials have not yet been configured. The `.env` file is excluded from Git using `.gitignore` to prevent sensitive information from being committed.

## Running the Backend

### 1. Activate the Virtual Environment

For Windows PowerShell:

`venv\Scripts\activate`

If PowerShell blocks activation, run:

`Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser`

Then activate the virtual environment again.

### 2. Install Dependencies

`pip install fastapi uvicorn`

### 3. Start the Server

`uvicorn backend.main:app --reload`

The backend will run at:

`http://127.0.0.1:8000`

## Swagger API Documentation

FastAPI provides interactive API documentation through Swagger UI.

Open:

`http://127.0.0.1:8000/docs`

The available endpoints can be viewed and tested directly from Swagger UI.

## Git and GitHub
Git has been configured for version control and the project is maintained in a central GitHub repository. The backend structure and API routes have been committed and pushed to GitHub for team collaboration.

The `.gitignore` file excludes:

.env
venv/
__pycache__/
*.pyc

## Current Progress
The FastAPI backend environment has been set up successfully. The modular backend structure has been created, including the main application, route files, service layer, environment configuration, and utility/model folders. The `/health` and `/identify` endpoints were tested successfully through the backend structure, and the privacy layer has now been added to protect caretaker data.

We also implemented the initial caretaker privacy flow: token generation, phone encryption, and secure caregiver reference handling. The dog registration endpoint has been updated to keep the dog location visible while returning only a caregiver token instead of the raw phone number.

The local Uvicorn server is running successfully, Swagger documentation is available, and the project is connected to GitHub for version control and collaboration.

## Current Status
The basic backend architecture and API structure are ready and tested locally. The current endpoints provide the foundation for integrating the remaining project modules. The privacy layer for caretaker information has been implemented as an initial version and is ready for further integration with the database and WhatsApp flow. The actual AI pipeline, full database persistence, WhatsApp credentials, and complete end-to-end processing are not yet connected.

## Next Steps
- Connect the video processing pipeline
- Connect automatic nose detection
- Connect image enhancement
- Connect ResNet50 feature extraction
- Connect the vector database
- Implement Cosine Similarity search
- Connect dog registration with the database
- Implement QR generation and fallback identification
- Configure WhatsApp Cloud API
- Connect the WhatsApp webhook
- Integrate all modules into one complete end-to-end workflow

## Planned Workflow

WhatsApp / User
        ↓
FastAPI Backend
        ↓
Video Processing
        ↓
Nose Detection
        ↓
Image Enhancement
        ↓
ResNet50 Feature Extraction
        ↓
Vector Database
        ↓
Cosine Similarity Search
        ↓
Dog ID / Registration Result
        ↓
WhatsApp Response

## Project Goal
The goal of the backend is to provide a central API layer that connects the Canine Nose-Print Recognition pipeline with the AI processing modules, database, QR fallback system, and WhatsApp interface, enabling the complete identification and registration workflow for street dogs.

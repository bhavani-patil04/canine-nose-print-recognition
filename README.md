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
| POST | `/register` | Handles new dog registration |
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
The FastAPI backend environment has been set up successfully. The modular backend structure has been created, including the main application, route files, service layer, environment configuration, and utility/model folders. The `/health` and `/identify` endpoints have been tested successfully using Swagger UI, and the remaining core endpoints have been created as the initial API structure.

The local Uvicorn server is running successfully, Swagger documentation is available, and the project is connected to GitHub for version control and collaboration.

## Current Status
The basic backend architecture and API structure are ready and tested locally. The current endpoints provide the foundation for integrating the remaining project modules. The actual AI pipeline, database connection, WhatsApp credentials, and complete end-to-end processing are not yet connected.

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

# Living Bridges - WhatsApp Integration

Official WhatsApp Cloud API backend & chatbot for the **Living Bridges** Stray Dog Tracking and Identification System.

---

## 🌟 Overview

The **Living Bridges WhatsApp Integration** allows community members and caregivers to:
- 🐕 **Identify stray dogs** by sending a photo or dog ID.
- 📋 **Report observations & status updates** (vaccinations, health conditions, locations).
- 🐾 **Book grooming, health checks, and vaccination appointments** via an interactive conversational flow and quick-reply buttons.
- 📍 **Submit geolocation and updates** directly through WhatsApp messages.
- 🖥️ **Admin Portal & API** for database synchronization and managing recorded profiles.

---

## 🚀 Features

- **Meta WhatsApp Cloud API Integration**: Handles real-time webhook events (`GET` verification and `POST` messaging).
- **Interactive Interactive/Button Messages**: Dynamic response menus, booking flows, and status prompts.
- **SQLite Dog Database**: Fast, embedded relational tracking of canine profiles, histories, vaccinations, and status records.
- **FastAPI Webhook Server**: High-performance asynchronous endpoint with automatic documentation via Swagger UI.

---

## 🛠️ Tech Stack

- **Framework**: FastAPI (Python 3.10+)
- **Server**: Uvicorn
- **HTTP Client**: HTTPX
- **Database**: SQLite3
- **External APIs**: Meta WhatsApp Business Cloud API

---

## ⚙️ Setup & Installation

### 1. Clone the repository
```bash
git clone https://github.com/sntharun/Living-Bridges.git
cd Living-Bridges
```

### 2. Create and activate a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env` and fill in your Meta Cloud API credentials:
```bash
cp .env.example .env
```

Edit `.env`:
```env
META_ACCESS_TOKEN=your_meta_access_token_here
META_PHONE_NUMBER_ID=your_meta_phone_number_id_here
META_VERIFY_TOKEN=living_bridges_secret_123
META_API_VERSION=v20.0
```

---

## 🏃 Running the Application

### Start the FastAPI Server
```bash
python3 -m uvicorn whatsapp:app --reload --port 8000
```

### Expose with ngrok (for Webhook testing)
```bash
ngrok http 8000
```

Configure your webhook callback URL in the [Meta Developer Portal](https://developers.facebook.com/apps/):
- **Callback URL**: `https://<your-ngrok-subdomain>.ngrok-free.app/webhook`
- **Verify Token**: `living_bridges_secret_123` (or the token specified in `.env`)
- **Webhook Fields**: `messages`

---

## 📖 API Endpoints

- `GET /` — Service health & dashboard
- `GET /webhook` — Meta Webhook verification handshake
- `POST /webhook` — Inbound WhatsApp messages, media, and button payloads
- `GET /docs` — Swagger UI API documentation

---

## 📄 License

This project is licensed under the MIT License.

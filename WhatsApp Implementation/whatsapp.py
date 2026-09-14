import sqlite3
import uuid
import os
import json
import logging
import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, PlainTextResponse, HTMLResponse
from pydantic import BaseModel

# ========================================================
# ENVIRONMENT & META WHATSAPP CLOUD API CONFIGURATION
# ========================================================

def load_env_file(filepath=".env"):
    """Reads .env file and sets environment variables."""
    env_path = filepath if os.path.isabs(filepath) else os.path.join(os.path.dirname(os.path.abspath(__file__)), filepath)
    if os.path.exists(env_path):
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ[k.strip()] = v.strip().strip('"').strip("'")

def get_meta_config():
    load_env_file(".env")
    return {
        "token": os.environ.get("META_ACCESS_TOKEN", ""),
        "phone_id": os.environ.get("META_PHONE_NUMBER_ID", ""),
        "verify_token": os.environ.get("META_VERIFY_TOKEN", "living_bridges_secret_123"),
        "version": os.environ.get("META_API_VERSION", "v20.0")
    }

load_env_file(".env")


app = FastAPI(title="Living Bridges Dog Tracker API")



# ========================================================
# DATABASE CONFIGURATION & INITIALIZATION
# ========================================================

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dogs.db")

def get_connection():
    """Returns a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    return conn

def init_db():
    """Initializes the database schema if it does not exist."""
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dogs (
            dog_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            area TEXT NOT NULL,
            medical_details TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()

# Initialize DB on module import
init_db()

# ========================================================
# STATE MANAGEMENT
# ========================================================

user_states = {}
user_registration = {}
user_update_dogs = {}

# ========================================================
# HELPER FUNCTIONS
# ========================================================

def main_menu() -> str:
    """Returns the formatted main menu string."""
    return (
        "🐾 *Living Bridges Dog Tracker*\n\n"
        "1️⃣ Register a new dog\n"
        "2️⃣ Update existing dog medical details\n"
        "3️⃣ View all registered dogs\n\n"
        "Please reply with *1*, *2*, or *3*."
    )

def valid_medical_details(text: str) -> bool:
    """Validates that medical details include vaccination or sterilization info."""
    if not text:
        return False
    lower = text.lower()
    return "vaccin" in lower or "steril" in lower

def confirmation_message(user_id: str) -> str:
    """Constructs the registration confirmation message for review."""
    data = user_registration.get(user_id, {})
    name = data.get("name", "N/A")
    area = data.get("area", "N/A")
    medical = data.get("medical_details", "N/A")

    return (
        "📋 *Registration Summary:*\n\n"
        f"🐶 *Name:* {name}\n"
        f"📍 *Area:* {area}\n"
        f"🏥 *Medical Details:*\n{medical}\n\n"
        "Reply with:\n"
        "• *1* or *Confirm* to save\n"
        "• *2* or *Change Name* to edit name\n"
        "• *3* or *Change Area* to edit area\n"
        "• *4* or *Change Medical* to edit medical details\n"
        "• *Cancel* to discard and return to menu"
    )

# ========================================================
# MAIN MESSAGE HANDLER
# ========================================================

def handle_message(user_id: str, text: str = "", message_type: str = "text") -> dict:
    """
    Handles incoming messages from WhatsApp users based on conversation state.
    
    Args:
        user_id (str): Unique identifier for the user (e.g. phone number).
        text (str): The text message sent by the user.
        message_type (str): Type of message ('text', 'image', 'video', etc.).
        
    Returns:
        dict: A dictionary containing the response payload, e.g. {"reply": ...}.
    """
    text = text.strip() if text else ""
    state = user_states.get(user_id, "menu")

    # Global reset command
    if text.lower() in ["menu", "restart", "start", "hi", "hello", "reset"] and state == "menu":
        user_states[user_id] = "menu"
        return {"reply": main_menu()}

    # ========================================================
    # MAIN MENU STATE
    # ========================================================
    if state == "menu":
        if text in ["1", "register", "register dog", "new dog"]:
            user_registration[user_id] = {}
            user_states[user_id] = "registration_name"
            return {
                "reply": (
                    "🐾 *New Dog Registration*\n\n"
                    "Please enter the dog's name (or a description if unnamed):\n\n"
                    "Example: Bruno"
                )
            }
        elif text in ["2", "update", "update dog", "update medical"]:
            user_states[user_id] = "update_dog_id"
            return {
                "reply": (
                    "🔍 *Update Dog Record*\n\n"
                    "Please enter the Dog ID:\n\n"
                    "Example:\n"
                    "DOG-4F0FB602"
                )
            }
        elif text in ["3", "view", "list", "view dogs"]:
            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("SELECT dog_id, name, area, medical_details FROM dogs ORDER BY created_at DESC LIMIT 10")
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return {
                    "reply": "ℹ️ No dogs registered yet.\n\n" + main_menu()
                }

            listing = "📋 *Registered Dogs (Recent 10):*\n\n"
            for row in rows:
                listing += f"🆔 *{row[0]}* - {row[1]} (📍 {row[2]})\n  🏥 {row[3]}\n\n"

            listing += main_menu()
            return {"reply": listing}
        else:
            return {
                "reply": (
                    "❌ Invalid choice. Please select an option from the menu.\n\n"
                    + main_menu()
                )
            }

    # ========================================================
    # REGISTRATION: NAME
    # ========================================================
    if state == "registration_name":
        if message_type in ["image", "video"]:
            return {"reply": "❌ Please enter the dog's name as text."}

        if not text:
            return {"reply": "❌ Name cannot be empty.\n\nPlease enter the dog's name."}

        if text.isdigit():
            return {"reply": "❌ Please enter a valid name (not only numbers)."}

        user_registration[user_id] = {"name": text}
        user_states[user_id] = "registration_area"

        return {
            "reply": (
                f"✅ Dog name set to *{text}*.\n\n"
                "📍 Now, please enter the area/location where the dog is located:\n\n"
                "Example: Mysore"
            )
        }

    # ========================================================
    # REGISTRATION: AREA
    # ========================================================
    if state == "registration_area":
        if message_type in ["image", "video"]:
            return {"reply": "❌ Please enter the area as text.\n\nExample: Mysore"}

        if not text:
            return {"reply": "❌ Area cannot be empty.\n\nPlease enter the area."}

        saved_name = user_registration.get(user_id, {}).get("name", "")
        if text.lower() == saved_name.lower():
            return {
                "reply": (
                    "❌ Area cannot be the same as the dog's name.\n\n"
                    "Please enter the actual area/location."
                )
            }

        if text.isdigit():
            return {"reply": "❌ Please enter a valid area/location."}

        user_registration[user_id]["area"] = text
        user_states[user_id] = "registration_medical"

        return {
            "reply": (
                f"✅ Area set to *{text}*.\n\n"
                "🏥 Please enter medical details:\n\n"
                "Example:\n"
                "Vaccination: Yes\n"
                "Sterilization: No"
            )
        }

    # ========================================================
    # REGISTRATION: MEDICAL
    # ========================================================
    if state == "registration_medical":
        if message_type in ["image", "video"]:
            return {
                "reply": (
                    "❌ Please enter medical details as text.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        if not text:
            return {"reply": "❌ Medical details cannot be empty."}

        if not valid_medical_details(text):
            return {
                "reply": (
                    "❌ Please include vaccination or sterilization details.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        user_registration[user_id]["medical_details"] = text
        user_states[user_id] = "registration_confirm"

        return {"reply": confirmation_message(user_id)}

    # ========================================================
    # REGISTRATION: CONFIRMATION
    # ========================================================
    if state == "registration_confirm":
        choice = text.lower()
        if choice in ["1", "confirm", "yes", "save"]:
            dog_data = user_registration.get(user_id, {})
            dog_id = f"DOG-{uuid.uuid4().hex[:8].upper()}"

            conn = get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO dogs (dog_id, name, area, medical_details)
                VALUES (?, ?, ?, ?)
            """, (
                dog_id,
                dog_data.get("name", "Unknown"),
                dog_data.get("area", "Unknown"),
                dog_data.get("medical_details", "Unknown")
            ))
            conn.commit()
            conn.close()

            user_registration.pop(user_id, None)
            user_states[user_id] = "menu"

            return {
                "reply": (
                    "🎉 *Dog registered successfully!*\n\n"
                    f"🆔 Dog ID: {dog_id}\n"
                    f"🐶 Name: {dog_data.get('name')}\n"
                    f"📍 Area: {dog_data.get('area')}\n"
                    f"🏥 Medical: {dog_data.get('medical_details')}\n\n"
                    + main_menu()
                )
            }
        elif choice in ["2", "change name", "name"]:
            user_states[user_id] = "registration_change_name"
            return {"reply": "✏️ Please enter the new name for the dog:"}
        elif choice in ["3", "change area", "area"]:
            user_states[user_id] = "registration_change_area"
            return {"reply": "✏️ Please enter the new area/location:"}
        elif choice in ["4", "change medical", "medical"]:
            user_states[user_id] = "registration_change_medical"
            return {
                "reply": (
                    "✏️ Please enter the new medical details:\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }
        elif choice in ["cancel", "exit", "discard"]:
            user_registration.pop(user_id, None)
            user_states[user_id] = "menu"
            return {
                "reply": "❌ Registration cancelled.\n\n" + main_menu()
            }
        else:
            return {
                "reply": (
                    "❌ Invalid option.\n\n"
                    + confirmation_message(user_id)
                )
            }

    # ========================================================
    # CHANGE NAME
    # ========================================================
    if state == "registration_change_name":
        if message_type in ["image", "video"]:
            return {"reply": "❌ Please enter the new name as text."}

        if not text:
            return {"reply": "❌ Name cannot be empty.\n\nPlease enter the name."}

        if text.isdigit():
            return {"reply": "❌ Please enter a valid name."}

        if user_id not in user_registration:
            user_registration[user_id] = {}
        user_registration[user_id]["name"] = text
        user_states[user_id] = "registration_confirm"

        return {"reply": confirmation_message(user_id)}

    # ========================================================
    # CHANGE AREA
    # ========================================================
    if state == "registration_change_area":

        if message_type in ["image", "video"]:
            return {
                "reply": (
                    "❌ Please enter the new area as text.\n\n"
                    "Example: Mysore"
                )
            }

        new_area = text

        if not new_area:
            return {
                "reply": (
                    "❌ Area cannot be empty.\n\n"
                    "Please enter the area."
                )
            }

        saved_name = user_registration.get(user_id, {}).get("name", "")

        if new_area.lower() == saved_name.lower():
            return {
                "reply": (
                    "❌ Area cannot be the same as "
                    "the dog's name.\n\n"
                    "Please enter the actual area/location."
                )
            }

        if new_area.isdigit():
            return {
                "reply": (
                    "❌ Please enter a valid area/location."
                )
            }

        if user_id not in user_registration:
            user_registration[user_id] = {}
        user_registration[user_id]["area"] = new_area

        user_states[user_id] = "registration_confirm"

        return {
            "reply": confirmation_message(user_id)
        }

    # ========================================================
    # CHANGE MEDICAL DETAILS
    # ========================================================
    if state == "registration_change_medical":

        if message_type in ["image", "video"]:
            return {
                "reply": (
                    "❌ Please enter medical details as text.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        new_medical = text

        if not new_medical:
            return {
                "reply": (
                    "❌ Medical details cannot be empty."
                )
            }

        if not valid_medical_details(new_medical):
            return {
                "reply": (
                    "❌ Please include vaccination "
                    "or sterilization details.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        if user_id not in user_registration:
            user_registration[user_id] = {}
        user_registration[user_id]["medical_details"] = new_medical

        user_states[user_id] = "registration_confirm"

        return {
            "reply": confirmation_message(user_id)
        }

    # ========================================================
    # UPDATE EXISTING DOG - DOG ID
    # ========================================================
    if state == "update_dog_id":

        dog_id = text.upper().strip()

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT dog_id, name, area, medical_details
            FROM dogs
            WHERE dog_id = ?
        """, (dog_id,))

        row = cursor.fetchone()

        conn.close()

        if row is None:
            return {
                "reply": (
                    "❌ Dog ID not found.\n\n"
                    "Please enter a valid Dog ID.\n\n"
                    "Example:\n"
                    "DOG-4F0FB602"
                )
            }

        user_update_dogs[user_id] = dog_id

        user_states[user_id] = "update_medical"

        return {
            "reply": (
                "✅ Dog found.\n\n"
                f"🐶 Name: {row[1]}\n"
                f"📍 Area: {row[2]}\n\n"
                "Please enter the new medical details.\n\n"
                "Example:\n"
                "Vaccination: Yes\n"
                "Sterilization: No"
            )
        }

    # ========================================================
    # UPDATE EXISTING DOG - MEDICAL DETAILS
    # ========================================================
    if state == "update_medical":

        if message_type in ["image", "video"]:
            return {
                "reply": (
                    "❌ Please enter medical details as text."
                )
            }

        medical_details = text

        if not medical_details:
            return {
                "reply": (
                    "❌ Medical details cannot be empty."
                )
            }

        if not valid_medical_details(medical_details):
            return {
                "reply": (
                    "❌ Please include vaccination "
                    "or sterilization details.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        dog_id = user_update_dogs.get(user_id)
        if not dog_id:
            user_states[user_id] = "menu"
            return {"reply": "❌ Session expired. Please start again.\n\n" + main_menu()}

        conn = get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            UPDATE dogs
            SET medical_details = ?
            WHERE dog_id = ?
        """, (
            medical_details,
            dog_id
        ))

        conn.commit()
        conn.close()

        user_update_dogs.pop(user_id, None)

        user_states[user_id] = "menu"

        return {
            "reply": (
                "✅ *Dog record updated successfully!*\n\n"
                f"🆔 Dog ID: {dog_id}\n"
                f"🏥 Medical Details: {medical_details}\n\n"
                + main_menu()
            )
        }

    # ========================================================
    # FINAL FALLBACK
    # ========================================================
    user_states[user_id] = "menu"

    return {
        "reply": (
            "❌ Something went wrong.\n\n"
            "Let's start again.\n\n"
            + main_menu()
        )
    }

# ========================================================
# FASTAPI WEBHOOK ENDPOINTS
# ========================================================

class MessagePayload(BaseModel):
    user_id: str
    text: str = ""
    message_type: str = "text"

WHATSAPP_UI_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Living Bridges Dog Tracker - WhatsApp</title>
    <link href="https://fonts.googleapis.com/css2?family=Segoe+UI:wght@400;500;600;700&family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
            font-family: 'Segoe UI', 'Inter', -apple-system, BlinkMacSystemFont, Roboto, sans-serif;
        }

        body {
            background: #111b21;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            padding: 10px;
        }

        .chat-app {
            width: 100%;
            max-width: 540px;
            height: 94vh;
            background: #0b141a;
            display: flex;
            flex-direction: column;
            border-radius: 16px;
            overflow: hidden;
            box-shadow: 0 12px 40px rgba(0, 0, 0, 0.6);
            border: 1px solid #222e35;
        }

        /* Top Header */
        .chat-header {
            background: #202c33;
            padding: 12px 16px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            color: #e9edef;
            border-bottom: 1px solid #222e35;
            z-index: 10;
        }

        .header-left {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .avatar {
            width: 44px;
            height: 44px;
            border-radius: 50%;
            background: linear-gradient(135deg, #00a884, #005c4b);
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 22px;
            box-shadow: 0 2px 8px rgba(0, 168, 132, 0.3);
        }

        .chat-title h3 {
            font-size: 16px;
            font-weight: 600;
            color: #e9edef;
            display: flex;
            align-items: center;
            gap: 6px;
        }

        .chat-title p {
            font-size: 12px;
            color: #8696a0;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .online-dot {
            width: 8px;
            height: 8px;
            background: #00a884;
            border-radius: 50%;
            display: inline-block;
        }

        .header-right {
            display: flex;
            align-items: center;
            gap: 8px;
        }

        .user-id-badge {
            background: #111b21;
            padding: 6px 10px;
            border-radius: 8px;
            font-size: 12px;
            color: #00a884;
            border: 1px solid #2a3942;
            display: flex;
            align-items: center;
            gap: 4px;
        }

        .user-id-badge input {
            background: transparent;
            border: none;
            color: #00a884;
            font-size: 12px;
            width: 95px;
            outline: none;
            font-weight: 500;
        }

        .reset-btn {
            background: #2a3942;
            border: none;
            color: #d1d7db;
            padding: 6px 12px;
            border-radius: 8px;
            font-size: 12px;
            cursor: pointer;
            transition: all 0.2s;
        }

        .reset-btn:hover {
            background: #ef4444;
            color: #fff;
        }

        /* Message Area */
        .chat-messages {
            flex: 1;
            padding: 16px;
            overflow-y: auto;
            background-color: #0b141a;
            background-image: radial-gradient(#182229 1px, transparent 1px);
            background-size: 20px 20px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }

        .chat-messages::-webkit-scrollbar {
            width: 6px;
        }

        .chat-messages::-webkit-scrollbar-thumb {
            background: #374248;
            border-radius: 3px;
        }

        .msg-container {
            display: flex;
            flex-direction: column;
            max-width: 82%;
            animation: fadeIn 0.25s ease-out;
        }

        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(6px); }
            to { opacity: 1; transform: translateY(0); }
        }

        .msg-container.bot {
            align-self: flex-start;
        }

        .msg-container.user {
            align-self: flex-end;
        }

        .message-bubble {
            padding: 10px 14px;
            border-radius: 12px;
            font-size: 14.5px;
            line-height: 1.45;
            position: relative;
            word-break: break-word;
            white-space: pre-wrap;
        }

        .bot .message-bubble {
            background: #202c33;
            color: #e9edef;
            border-top-left-radius: 2px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }

        .user .message-bubble {
            background: #005c4b;
            color: #e9edef;
            border-top-right-radius: 2px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.3);
        }

        .msg-time {
            font-size: 11px;
            color: #8696a0;
            float: right;
            margin-left: 12px;
            margin-top: 4px;
            display: flex;
            align-items: center;
            gap: 3px;
        }

        .user .msg-time {
            color: #8bb7a8;
        }

        .check-icon {
            color: #53bdeb;
            font-size: 12px;
        }

        /* Quick Action Chips */
        .quick-actions {
            display: flex;
            flex-wrap: wrap;
            gap: 6px;
            margin-top: 8px;
            align-self: flex-start;
        }

        .quick-btn {
            background: #111b21;
            border: 1px solid #00a884;
            color: #00a884;
            padding: 6px 12px;
            border-radius: 16px;
            font-size: 13px;
            cursor: pointer;
            transition: all 0.2s;
            font-weight: 500;
        }

        .quick-btn:hover {
            background: #00a884;
            color: #111b21;
        }

        /* Input Area */
        .chat-input-bar {
            background: #202c33;
            padding: 10px 14px;
            display: flex;
            align-items: center;
            gap: 10px;
            border-top: 1px solid #222e35;
        }

        .input-wrapper {
            flex: 1;
            background: #2a3942;
            border-radius: 24px;
            padding: 8px 16px;
            display: flex;
            align-items: center;
        }

        .input-wrapper input {
            width: 100%;
            background: transparent;
            border: none;
            color: #d1d7db;
            font-size: 15px;
            outline: none;
        }

        .input-wrapper input::placeholder {
            color: #8696a0;
        }

        .send-btn {
            width: 42px;
            height: 42px;
            border-radius: 50%;
            background: #00a884;
            border: none;
            color: #111b21;
            display: flex;
            align-items: center;
            justify-content: center;
            cursor: pointer;
            transition: transform 0.15s, background-color 0.2s;
            flex-shrink: 0;
        }

        .send-btn:hover {
            background: #06cf9c;
            transform: scale(1.05);
        }

        .send-btn:active {
            transform: scale(0.95);
        }

        .send-btn svg {
            width: 20px;
            height: 20px;
            fill: #111b21;
            margin-left: 2px;
        }

        .typing-indicator {
            display: none;
            font-size: 12px;
            color: #00a884;
            padding: 4px 16px;
            font-style: italic;
        }
    </style>
</head>
<body>

<div class="chat-app">
    <!-- Header -->
    <div class="chat-header">
        <div class="header-left">
            <div class="avatar">🐾</div>
            <div class="chat-title">
                <h3>Living Bridges Dog Tracker</h3>
                <p><span class="online-dot"></span> online • verified bot</p>
            </div>
        </div>
        <div class="header-right">
            <div class="user-id-badge" title="Simulated WhatsApp Phone Number">
                📱 <input id="userIdInput" value="+919876543210" title="Change test phone number">
            </div>
            <button class="reset-btn" onclick="resetChat()" title="Reset session">Restart</button>
        </div>
    </div>

    <!-- Messages List -->
    <div class="chat-messages" id="chatMessages">
        <!-- Messages rendered here -->
    </div>

    <div class="typing-indicator" id="typingIndicator">Bot is typing...</div>

    <!-- Input Footer -->
    <div class="chat-input-bar">
        <div class="input-wrapper">
            <input type="text" id="messageInput" placeholder="Type a message (e.g. 1, 2, 3, or dog name)..." autocomplete="off">
        </div>
        <button class="send-btn" id="sendBtn" onclick="sendMessage()">
            <svg viewBox="0 0 24 24">
                <path d="M2.01 21L23 12 2.01 3 2 10l15 2-15 2z"></path>
            </svg>
        </button>
    </div>
</div>

<script>
    const chatMessages = document.getElementById('chatMessages');
    const messageInput = document.getElementById('messageInput');
    const userIdInput = document.getElementById('userIdInput');
    const typingIndicator = document.getElementById('typingIndicator');

    function formatWhatsAppText(text) {
        if (!text) return '';
        // Escape HTML
        let html = text
            .replace(/&/g, "&amp;")
            .replace(/</g, "&lt;")
            .replace(/>/g, "&gt;");

        // Format *bold*
        html = html.replace(/\\*([^*\\n]+)\\*/g, "<strong>$1</strong>");
        // Format _italics_
        html = html.replace(/_([^_\\n]+)_/g, "<em>$1</em>");
        // Format ~strike~
        html = html.replace(/~([^~\\n]+)~/g, "<del>$1</del>");
        // Format ```code```
        html = html.replace(/```([^`]+)```/g, "<code>$1</code>");

        return html;
    }

    function getCurrentTime() {
        const now = new Date();
        return now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    }

    function appendMessage(text, sender = 'bot') {
        const msgContainer = document.createElement('div');
        msgContainer.className = `msg-container ${sender}`;

        const bubble = document.createElement('div');
        bubble.className = 'message-bubble';
        
        const formatted = formatWhatsAppText(text);
        const timeHtml = sender === 'user' 
            ? `<span class="msg-time">${getCurrentTime()} <span class="check-icon">✓✓</span></span>`
            : `<span class="msg-time">${getCurrentTime()}</span>`;

        bubble.innerHTML = formatted + timeHtml;
        msgContainer.appendChild(bubble);

        // Detect quick buttons from bot replies
        if (sender === 'bot') {
            const quickActions = document.createElement('div');
            quickActions.className = 'quick-actions';

            if (text.includes("1️⃣ Register") && text.includes("2️⃣ Update")) {
                quickActions.innerHTML = `
                    <button class="quick-btn" onclick="sendQuick('1')">1️⃣ Register Dog</button>
                    <button class="quick-btn" onclick="sendQuick('2')">2️⃣ Update Details</button>
                    <button class="quick-btn" onclick="sendQuick('3')">3️⃣ View Dogs</button>
                `;
            } else if (text.includes("Registration Summary")) {
                quickActions.innerHTML = `
                    <button class="quick-btn" onclick="sendQuick('1')">✅ Confirm (1)</button>
                    <button class="quick-btn" onclick="sendQuick('2')">✏️ Edit Name (2)</button>
                    <button class="quick-btn" onclick="sendQuick('3')">📍 Edit Area (3)</button>
                    <button class="quick-btn" onclick="sendQuick('4')">🏥 Edit Medical (4)</button>
                    <button class="quick-btn" onclick="sendQuick('Cancel')">❌ Cancel</button>
                `;
            }

            if (quickActions.children.length > 0) {
                msgContainer.appendChild(quickActions);
            }
        }

        chatMessages.appendChild(msgContainer);
        chatMessages.scrollTop = chatMessages.scrollHeight;
    }

    async function sendMessage(customText = null) {
        const text = customText !== null ? customText : messageInput.value.trim();
        if (!text) return;

        const userId = userIdInput.value.trim() || "+919876543210";

        appendMessage(text, 'user');
        if (customText === null) {
            messageInput.value = '';
        }

        typingIndicator.style.display = 'block';

        try {
            const res = await fetch('/webhook', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    user_id: userId,
                    text: text,
                    message_type: 'text'
                })
            });

            const data = await res.json();
            typingIndicator.style.display = 'none';

            if (data && data.reply) {
                appendMessage(data.reply, 'bot');
            } else {
                appendMessage("❌ No response from bot server.", 'bot');
            }
        } catch (err) {
            typingIndicator.style.display = 'none';
            appendMessage("❌ Connection error: Could not reach server.", 'bot');
        }
    }

    function sendQuick(val) {
        sendMessage(val);
    }

    async function resetChat() {
        chatMessages.innerHTML = '';
        await sendMessage('hi');
    }

    messageInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter') {
            sendMessage();
        }
    });

    // Send initial greeting on page load
    window.addEventListener('DOMContentLoaded', () => {
        sendMessage('hi');
    });
</script>

</body>
</html>
"""

@app.get("/", response_class=HTMLResponse)
def root():
    return HTMLResponse(content=WHATSAPP_UI_HTML)


@app.post("/webhook")
def webhook(payload: MessagePayload):
    response = handle_message(
        user_id=payload.user_id,
        text=payload.text,
        message_type=payload.message_type
    )
    return response

# ========================================================
# META WHATSAPP CLOUD API MESSAGING FUNCTIONS
# ========================================================

async def send_meta_whatsapp_message(to_number: str, message_text: str, phone_number_id: str = None) -> bool:
    """
    Sends a WhatsApp message using Meta Cloud Graph API.
    """
    config = get_meta_config()
    phone_id = phone_number_id or config["phone_id"]
    access_token = config["token"]
    api_version = config["version"]

    if not access_token or not phone_id:
        print("⚠️ [Meta API] META_ACCESS_TOKEN or META_PHONE_NUMBER_ID is not configured. Reply not sent to WhatsApp.")
        print(f"   (Message for {to_number}: {message_text[:60]}...)")
        return False

    url = f"https://graph.facebook.com/{api_version}/{phone_id}/messages"
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json"
    }
    payload = {
        "messaging_product": "whatsapp",
        "recipient_type": "individual",
        "to": to_number,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message_text
        }
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(url, json=payload, headers=headers)
            if resp.status_code == 200:
                print(f"✅ [Meta API] Message sent successfully to {to_number}")
                return True
            else:
                print(f"❌ [Meta API] Error sending message ({resp.status_code}): {resp.text}")
                return False
    except Exception as e:
        print(f"❌ [Meta API] Exception while sending message: {e}")
        return False


@app.get("/whatsapp")
def meta_verify_webhook(request: Request):
    """
    Webhook verification endpoint for Meta WhatsApp Cloud API.
    Meta sends a GET request with hub.mode, hub.verify_token, and hub.challenge.
    """
    config = get_meta_config()
    params = request.query_params
    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")
    
    expected_token = config["verify_token"]

    if mode == "subscribe" and token == expected_token:
        print("✅ [Meta API] Webhook verified successfully by Meta!")
        return PlainTextResponse(content=challenge, status_code=200)
    
    print(f"❌ [Meta API] Webhook verification failed. Received token: {token}, Expected: {expected_token}")
    return PlainTextResponse(content="Verification token mismatch", status_code=403)



@app.post("/whatsapp")
async def meta_whatsapp_webhook(request: Request):
    """
    Webhook receiver for incoming Meta WhatsApp Cloud API messages.
    """
    try:
        data = await request.json()
    except Exception:
        return JSONResponse(content={"status": "invalid json"}, status_code=400)

    entries = data.get("entry", [])
    for entry in entries:
        changes = entry.get("changes", [])
        for change in changes:
            value = change.get("value", {})
            phone_number_id = value.get("metadata", {}).get("phone_number_id")

            # Check if there are messages (ignore status updates like 'sent' or 'read')
            if "messages" in value:
                for message in value["messages"]:
                    sender_id = message.get("from")  # User phone number (e.g. '919876543210')
                    msg_type = message.get("type", "text")
                    
                    text_content = ""
                    if msg_type == "text":
                        text_content = message.get("text", {}).get("body", "")
                    elif msg_type == "interactive":
                        interactive = message.get("interactive", {})
                        int_type = interactive.get("type")
                        if int_type == "button_reply":
                            text_content = interactive.get("button_reply", {}).get("title", "")
                        elif int_type == "list_reply":
                            text_content = interactive.get("list_reply", {}).get("title", "")
                    elif msg_type in ["image", "document", "audio", "video"]:
                        text_content = f"[{msg_type}]"
                    
                    if sender_id and text_content:
                        print(f"📩 [Meta API] Received message from {sender_id}: {text_content}")
                        bot_response = handle_message(
                            user_id=sender_id,
                            text=text_content,
                            message_type=msg_type
                        )
                        
                        reply_text = bot_response.get("reply", "")
                        if reply_text:
                            await send_meta_whatsapp_message(
                                to_number=sender_id,
                                message_text=reply_text,
                                phone_number_id=phone_number_id
                            )

    return JSONResponse(content={"status": "ok"}, status_code=200)

# ========================================================
# CLI INTERACTIVE TEST RUNNER
# ========================================================
if __name__ == "__main__":
    print("=" * 60)
    print("🐾 Living Bridges WhatsApp Bot (Interactive Terminal Mode)")
    print("Type your messages below. Type 'exit' to quit.")
    print("=" * 60)
    
    test_user_id = "+1234567890"
    init_res = handle_message(test_user_id, "hi")
    print(f"\nBot:\n{init_res['reply']}\n")

    while True:
        try:
            user_input = input("You: ")
            if user_input.strip().lower() in ["exit", "quit"]:
                print("Goodbye!")
                break
            
            response = handle_message(test_user_id, user_input)
            print(f"\nBot:\n{response['reply']}\n")
        except (KeyboardInterrupt, EOFError):
            print("\nSession ended.")
            break
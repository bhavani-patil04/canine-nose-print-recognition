from fastapi import FastAPI, Request, Query
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel
import sqlite3
import os
import uuid
import qrcode
import requests


# ============================================================
# APP
# ============================================================

app = FastAPI(title="Living Bridges - WhatsApp Dog Registration")


# ============================================================
# META WHATSAPP SETTINGS
# ============================================================

VERIFY_TOKEN = "livingbridges123"

# IMPORTANT:
# Paste your Meta-generated access token here.
# DO NOT SHARE THIS TOKEN WITH ANYONE.
WHATSAPP_ACCESS_TOKEN = "EAAewbDdO3D8BSvyxjl00kjRZCnPZCiJxCD8fGUvMBXsYWZAzLXF2CmBOhTqnJXZANvg9ZBe56ZC2HGSb9M2mMf2sifvrNKZB1cKfyQm841xPm84JuUcAc2iRxN678KalRKkkk092EPNlECjuZAs1WG39LkQZBBR2PXMPE3AXIb4LvRnSb62jaEeWWbuyFYpexMzpZAmo0DlW0amju7nV5rQvYNIhq8rCnCZCqFuGGPbY32BMNkm1bcZCcVliyJUIPFExIhbSisaGqDZCxrRWFPr4nZBfG6aWNVD5kjEcO0xAZDZD"

# Your Meta WhatsApp Test Phone Number ID
PHONE_NUMBER_ID = "1249526271585172"

# Meta Graph API version
GRAPH_API_VERSION = "v20.0"

WHATSAPP_API_URL = (
    f"https://graph.facebook.com/"
    f"{GRAPH_API_VERSION}/"
    f"{PHONE_NUMBER_ID}/messages"
)


# ============================================================
# WEBHOOK VERIFICATION
# ============================================================

@app.get("/webhook", response_class=PlainTextResponse)
def verify_webhook(
    hub_mode: str = Query(None, alias="hub.mode"),
    hub_verify_token: str = Query(None, alias="hub.verify_token"),
    hub_challenge: str = Query(None, alias="hub.challenge")
):

    if (
        hub_mode == "subscribe"
        and hub_verify_token == VERIFY_TOKEN
        and hub_challenge
    ):
        return hub_challenge

    return "Verification failed"


# ============================================================
# DATABASE
# ============================================================

DB_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "dogs.db"
)

print("Database path:", DB_PATH)


def get_connection():

    return sqlite3.connect(DB_PATH)


def create_database():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dogs (
            dog_id TEXT PRIMARY KEY,
            name TEXT,
            area TEXT,
            medical_details TEXT
        )
    """)

    conn.commit()
    conn.close()


create_database()


# ============================================================
# TEMPORARY USER STATES
# ============================================================

user_states = {}

user_registration = {}

user_update_dogs = {}


# ============================================================
# REQUEST MODEL FOR LOCAL TESTING
# ============================================================

class WhatsAppMessage(BaseModel):

    user_id: str

    message_text: str = ""

    message_type: str = "text"


# ============================================================
# BASIC ROUTES
# ============================================================

@app.get("/")
def home():

    return {
        "message": "Living Bridges WhatsApp Integration API is running"
    }


# ============================================================
# GET ALL DOGS
# ============================================================

@app.get("/dogs")
def get_all_dogs():

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        SELECT dog_id, name, area, medical_details
        FROM dogs
    """)

    rows = cursor.fetchall()

    conn.close()

    dogs = []

    for row in rows:

        dogs.append({
            "dog_id": row[0],
            "name": row[1],
            "area": row[2],
            "medical_details": row[3]
        })

    return dogs


# ============================================================
# GET ONE DOG
# ============================================================

@app.get("/dog/{dog_id}")
def get_dog(dog_id: str):

    dog_id = dog_id.strip().upper()

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
            "error": "Dog not found"
        }

    return {
        "dog_id": row[0],
        "name": row[1],
        "area": row[2],
        "medical_details": row[3]
    }


# ============================================================
# QR CODE GENERATOR
# ============================================================

def generate_qr(dog_id: str):

    qr_folder = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "qr_codes"
    )

    os.makedirs(qr_folder, exist_ok=True)

    qr = qrcode.QRCode(
        version=1,
        box_size=10,
        border=4
    )

    qr.add_data(dog_id)

    qr.make(fit=True)

    img = qr.make_image()

    file_path = os.path.join(
        qr_folder,
        f"{dog_id}.png"
    )

    img.save(file_path)

    return file_path


# ============================================================
# MAIN MENU
# ============================================================

def main_menu():

    return (
        "🐶 *Living Bridges*\n\n"
        "Please choose an option:\n\n"
        "1️⃣ Identify an Unknown Dog\n"
        "2️⃣ Register a Known New Dog\n"
        "3️⃣ Update Existing Dog Record\n\n"
        "Reply with 1, 2, or 3."
    )


# ============================================================
# CONFIRMATION MESSAGE
# ============================================================

def confirmation_message(user_id: str):

    data = user_registration[user_id]

    return (
        "📋 *Please confirm the dog's details*\n\n"

        f"🐶 Name: {data['name']}\n"
        f"📍 Area: {data['area']}\n"
        f"🏥 Medical Details: {data['medical_details']}\n\n"

        "Choose an option:\n\n"

        "1️⃣ Confirm & Register\n"
        "2️⃣ Change Name\n"
        "3️⃣ Change Area\n"
        "4️⃣ Change Medical Details\n"
        "5️⃣ Start Again"
    )


# ============================================================
# VALIDATE MEDICAL DETAILS
# ============================================================

def valid_medical_details(text: str):

    text_lower = text.lower()

    if (
        "vaccination" not in text_lower
        and "sterilization" not in text_lower
    ):
        return False

    return True


# ============================================================
# SEND TEXT MESSAGE TO WHATSAPP
# ============================================================

def send_whatsapp_message(
    recipient_phone: str,
    message_text: str
):

    headers = {
        "Authorization": f"Bearer {WHATSAPP_ACCESS_TOKEN}",
        "Content-Type": "application/json"
    }

    payload = {
        "messaging_product": "whatsapp",
        "to": recipient_phone,
        "type": "text",
        "text": {
            "preview_url": False,
            "body": message_text
        }
    }

    try:

        response = requests.post(
            WHATSAPP_API_URL,
            headers=headers,
            json=payload,
            timeout=20
        )

        print("WhatsApp API status:", response.status_code)
        print("WhatsApp API response:", response.text)

        return response.ok

    except Exception as error:

        print("WhatsApp API error:", error)

        return None


# ============================================================
# PROCESS MESSAGE
# ============================================================

def process_message(
    user_id: str,
    text: str,
    message_type: str
):

    user_id = user_id.strip()

    text = text.strip()

    message_type = message_type.lower().strip()

    state = user_states.get(
        user_id,
        "menu"
    )


    # ========================================================
    # START / MENU
    # ========================================================

    if text.lower() in [
        "hi",
        "hello",
        "start",
        "menu"
    ]:

        user_states[user_id] = "menu"

        return {
            "reply": main_menu()
        }


    # ========================================================
    # MAIN MENU
    # ========================================================

    if state == "menu":

        # ----------------------------------------------------
        # OPTION 1
        # ----------------------------------------------------

        if text == "1":

            user_states[user_id] = "identification"

            return {
                "reply": (
                    "🔎 *Identify an Unknown Dog*\n\n"
                    "Please send the dog's nose photo "
                    "or a 2-second nose video.\n\n"
                    "📷 Make sure the nose is clearly visible."
                )
            }


        # ----------------------------------------------------
        # OPTION 2
        # ----------------------------------------------------

        elif text == "2":

            user_registration[user_id] = {
                "name": None,
                "area": None,
                "medical_details": None
            }

            user_states[user_id] = "registration_nose"

            return {
                "reply": (
                    "📝 *Register a Known New Dog*\n\n"
                    "Please send the dog's nose photo "
                    "or a 2-second nose video."
                )
            }


        # ----------------------------------------------------
        # OPTION 3
        # ----------------------------------------------------

        elif text == "3":

            user_states[user_id] = "update_dog_id"

            return {
                "reply": (
                    "🔄 *Update Existing Dog Record*\n\n"
                    "Please enter the Dog ID.\n\n"
                    "Example:\n"
                    "DOG-4F0FB602"
                )
            }


        else:

            return {
                "reply": (
                    "❌ Invalid option.\n\n"
                    + main_menu()
                )
            }


    # ========================================================
    # IDENTIFICATION
    # ========================================================

    if state == "identification":

        if message_type in [
            "image",
            "video"
        ]:

            user_states[user_id] = "menu"

            return {
                "reply": (
                    "📷 Nose media received.\n\n"
                    "🤖 AI nose-print recognition will be "
                    "connected here.\n\n"
                    "For now, this is the prototype placeholder.\n\n"
                    + main_menu()
                )
            }

        return {
            "reply": (
                "❌ Please send a nose photo "
                "or a 2-second nose video."
            )
        }


    # ========================================================
    # REGISTRATION - NOSE
    # ========================================================

    if state == "registration_nose":

        if message_type in [
            "image",
            "video"
        ]:

            user_states[user_id] = "registration_name"

            return {
                "reply": (
                    "✅ Nose photo/video received.\n\n"
                    "Now enter the dog's name.\n\n"
                    "Example: Bruno"
                )
            }

        return {
            "reply": (
                "❌ Please send a nose photo "
                "or a 2-second nose video."
            )
        }


    # ========================================================
    # REGISTRATION - NAME
    # ========================================================

    if state == "registration_name":

        if message_type in [
            "image",
            "video"
        ]:

            return {
                "reply": (
                    "❌ Please enter the dog's name as text.\n\n"
                    "Example: Bruno"
                )
            }

        name = text

        if not name:

            return {
                "reply": (
                    "❌ Name cannot be empty.\n\n"
                    "Please enter the dog's name."
                )
            }

        if name.isdigit():

            return {
                "reply": (
                    "❌ Please enter a valid dog name.\n\n"
                    "Example: Bruno"
                )
            }

        user_registration[user_id]["name"] = name

        user_states[user_id] = "registration_area"

        return {
            "reply": (
                "✅ Name saved.\n\n"
                "Now enter the dog's area/location.\n\n"
                "Example: Mysore"
            )
        }


    # ========================================================
    # REGISTRATION - AREA
    # ========================================================

    if state == "registration_area":

        if message_type in [
            "image",
            "video"
        ]:

            return {
                "reply": (
                    "❌ Please enter the area/location as text.\n\n"
                    "Example: Mysore"
                )
            }

        area = text

        if not area:

            return {
                "reply": (
                    "❌ Area cannot be empty.\n\n"
                    "Please enter the dog's area/location."
                )
            }

        saved_name = user_registration[user_id]["name"]

        if area.lower() == saved_name.lower():

            return {
                "reply": (
                    "❌ Area cannot be the same as "
                    "the dog's name.\n\n"
                    f"Dog name: {saved_name}\n\n"
                    "Please enter the actual area/location.\n\n"
                    "Example: Mysore"
                )
            }

        if area.isdigit():

            return {
                "reply": (
                    "❌ Please enter a valid area/location.\n\n"
                    "Example: Mysore"
                )
            }

        user_registration[user_id]["area"] = area

        user_states[user_id] = "registration_medical"

        return {
            "reply": (
                "✅ Area saved.\n\n"
                "Now enter medical details.\n\n"
                "Example:\n"
                "Vaccination: Yes\n"
                "Sterilization: No"
            )
        }


    # ========================================================
    # REGISTRATION - MEDICAL
    # ========================================================

    if state == "registration_medical":

        if message_type in [
            "image",
            "video"
        ]:

            return {
                "reply": (
                    "❌ Please enter the medical details as text.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        medical_details = text

        if not medical_details:

            return {
                "reply": (
                    "❌ Medical details cannot be empty.\n\n"
                    "Please enter vaccination and "
                    "sterilization details."
                )
            }

        if not valid_medical_details(
            medical_details
        ):

            return {
                "reply": (
                    "❌ Please provide medical details.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        user_registration[user_id][
            "medical_details"
        ] = medical_details

        user_states[user_id] = "registration_confirm"

        return {
            "reply": confirmation_message(user_id)
        }


    # ========================================================
    # REGISTRATION - CONFIRMATION
    # ========================================================

    if state == "registration_confirm":

        choice = text


        # ----------------------------------------------------
        # CONFIRM
        # ----------------------------------------------------

        if choice == "1":

            data = user_registration[user_id]

            dog_id = (
                "DOG-"
                + uuid.uuid4().hex[:8].upper()
            )

            conn = get_connection()
            cursor = conn.cursor()

            cursor.execute("""
                INSERT INTO dogs
                (dog_id, name, area, medical_details)
                VALUES (?, ?, ?, ?)
            """, (
                dog_id,
                data["name"],
                data["area"],
                data["medical_details"]
            ))

            conn.commit()
            conn.close()

            qr_path = generate_qr(dog_id)

            dog_name = data["name"]
            dog_area = data["area"]

            del user_registration[user_id]

            user_states[user_id] = "menu"

            return {
                "reply": (
                    "✅ *Dog registered successfully!*\n\n"
                    f"🐶 Dog Name: {dog_name}\n"
                    f"📍 Area: {dog_area}\n"
                    f"🆔 Dog ID: {dog_id}\n\n"
                    "📱 QR code has been generated.\n\n"
                    + main_menu()
                ),
                "dog_id": dog_id,
                "qr_path": qr_path
            }


        # ----------------------------------------------------
        # CHANGE NAME
        # ----------------------------------------------------

        elif choice == "2":

            user_states[user_id] = (
                "registration_change_name"
            )

            return {
                "reply": (
                    "✏️ *Change Name*\n\n"
                    "Enter the new dog name.\n\n"
                    "Example: Bruno"
                )
            }


        # ----------------------------------------------------
        # CHANGE AREA
        # ----------------------------------------------------

        elif choice == "3":

            user_states[user_id] = (
                "registration_change_area"
            )

            return {
                "reply": (
                    "✏️ *Change Area*\n\n"
                    "Enter the new area/location.\n\n"
                    "Example: Mysore"
                )
            }


        # ----------------------------------------------------
        # CHANGE MEDICAL
        # ----------------------------------------------------

        elif choice == "4":

            user_states[user_id] = (
                "registration_change_medical"
            )

            return {
                "reply": (
                    "✏️ *Change Medical Details*\n\n"
                    "Enter the new medical details.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }


        # ----------------------------------------------------
        # START AGAIN
        # ----------------------------------------------------

        elif choice == "5":

            user_registration[user_id] = {
                "name": None,
                "area": None,
                "medical_details": None
            }

            user_states[user_id] = "registration_nose"

            return {
                "reply": (
                    "🔄 *Let's start again.*\n\n"
                    "Please send the dog's nose photo "
                    "or a 2-second nose video."
                )
            }


        else:

            return {
                "reply": (
                    "❌ Invalid option.\n\n"
                    "Please choose:\n\n"
                    "1️⃣ Confirm & Register\n"
                    "2️⃣ Change Name\n"
                    "3️⃣ Change Area\n"
                    "4️⃣ Change Medical Details\n"
                    "5️⃣ Start Again"
                )
            }


    # ========================================================
    # CHANGE NAME
    # ========================================================

    if state == "registration_change_name":

        if message_type in [
            "image",
            "video"
        ]:

            return {
                "reply": (
                    "❌ Please enter the new dog name as text.\n\n"
                    "Example: Bruno"
                )
            }

        new_name = text

        if not new_name:

            return {
                "reply": (
                    "❌ Name cannot be empty.\n\n"
                    "Please enter the new dog name."
                )
            }

        if new_name.isdigit():

            return {
                "reply": (
                    "❌ Please enter a valid dog name."
                )
            }

        user_registration[user_id][
            "name"
        ] = new_name

        user_states[user_id] = (
            "registration_confirm"
        )

        return {
            "reply": confirmation_message(user_id)
        }


    # ========================================================
    # CHANGE AREA
    # ========================================================

    if state == "registration_change_area":

        if message_type in [
            "image",
            "video"
        ]:

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

        saved_name = user_registration[user_id]["name"]

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

        user_registration[user_id][
            "area"
        ] = new_area

        user_states[user_id] = (
            "registration_confirm"
        )

        return {
            "reply": confirmation_message(user_id)
        }


    # ========================================================
    # CHANGE MEDICAL DETAILS
    # ========================================================

    if state == "registration_change_medical":

        if message_type in [
            "image",
            "video"
        ]:

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

        if not valid_medical_details(
            new_medical
        ):

            return {
                "reply": (
                    "❌ Please include vaccination "
                    "or sterilization details.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        user_registration[user_id][
            "medical_details"
        ] = new_medical

        user_states[user_id] = (
            "registration_confirm"
        )

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
    # UPDATE EXISTING DOG - MEDICAL
    # ========================================================

    if state == "update_medical":

        if message_type in [
            "image",
            "video"
        ]:

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

        if not valid_medical_details(
            medical_details
        ):

            return {
                "reply": (
                    "❌ Please include vaccination "
                    "or sterilization details.\n\n"
                    "Example:\n"
                    "Vaccination: Yes\n"
                    "Sterilization: No"
                )
            }

        dog_id = user_update_dogs[user_id]

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

        del user_update_dogs[user_id]

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


# ============================================================
# META WHATSAPP WEBHOOK
# ============================================================

@app.post("/webhook")
async def whatsapp_webhook(request: Request):
    print("POST /webhook received")
    try:

        payload = await request.json()

        print("\n==============================")
        print("META WEBHOOK RECEIVED")
        print("==============================")
        print(payload)


        # ----------------------------------------------------
        # CHECK BASIC PAYLOAD
        # ----------------------------------------------------

        if payload.get("object") != "whatsapp_business_account":

            return {
                "status": "ignored"
            }


        entries = payload.get("entry", [])

        for entry in entries:

            changes = entry.get("changes", [])

            for change in changes:

                value = change.get("value", {})

                messages = value.get(
                    "messages",
                    []
                )


                # ------------------------------------------------
                # IGNORE STATUS EVENTS
                # ------------------------------------------------

                if not messages:

                    print(
                        "No incoming message. "
                        "Probably a status event."
                    )

                    continue


                # ------------------------------------------------
                # PROCESS EACH MESSAGE
                # ------------------------------------------------

                for incoming_message in messages:

                    sender = incoming_message.get(
                        "from"
                    )

                    message_type = incoming_message.get(
                        "type",
                        "text"
                    )


                    # --------------------------------------------
                    # TEXT MESSAGE
                    # --------------------------------------------

                    if message_type == "text":

                        text_body = (
                            incoming_message
                            .get("text", {})
                            .get("body", "")
                        )


                    # --------------------------------------------
                    # IMAGE MESSAGE
                    # --------------------------------------------

                    elif message_type == "image":

                        text_body = ""

                    # --------------------------------------------
                    # VIDEO MESSAGE
                    # --------------------------------------------

                    elif message_type == "video":

                        text_body = ""

                    # --------------------------------------------
                    # OTHER MESSAGE
                    # --------------------------------------------

                    else:

                        text_body = ""


                    print(
                        "Sender:",
                        sender
                    )

                    print(
                        "Type:",
                        message_type
                    )

                    print(
                        "Text:",
                        text_body
                    )


                    # --------------------------------------------
                    # PROCESS USING EXISTING FLOW
                    # --------------------------------------------

                    result = process_message(
                        user_id=sender,
                        text=text_body,
                        message_type=message_type
                    )


                    reply = result.get(
                        "reply",
                        ""
                    )


                    print(
                        "Generated reply:",
                        reply
                    )


                    # --------------------------------------------
                    # SEND REPLY BACK TO WHATSAPP
                    # --------------------------------------------

                    if reply:

                        send_whatsapp_message(
                            recipient_phone=sender,
                            message_text=reply
                        )


        return {
            "status": "ok"
        }


    except Exception as error:

        print(
            "Webhook processing error:",
            error
        )

        return {
            "status": "error",
            "message": str(error)
        }


# ============================================================
# LOCAL TEST WEBHOOK
# ============================================================

@app.post("/test-message")
def test_message(message: WhatsAppMessage):

    result = process_message(
        user_id=message.user_id,
        text=message.message_text,
        message_type=message.message_type
    )

    return result

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
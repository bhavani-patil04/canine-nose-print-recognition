from fastapi import FastAPI, Request
from fastapi.responses import PlainTextResponse
from dotenv import load_dotenv
import os

from whatsapp import send_whatsapp_message

from state import (
    user_states,
    user_registration,
    user_update_dogs
)

from registration import (
    start_registration,
    handle_registration_nose,
    handle_registration_name,
    handle_registration_area,
    handle_registration_medical,
    confirmation_message,
    change_area,
    change_medical
)


load_dotenv()

app = FastAPI()

VERIFY_TOKEN = os.getenv("VERIFY_TOKEN")


# =========================================================
# MAIN MENU
# =========================================================

def main_menu():

    return (
        "🐶 *Welcome to Canine ID Bot!*\n\n"
        "Please select an option:\n\n"

        "1️⃣ Identify an Unknown Dog\n"
        "2️⃣ Register a Known New Dog\n"
        "3️⃣ Update Existing Dog Record\n\n"

        "Reply with 1, 2 or 3."
    )


# =========================================================
# WEBHOOK VERIFICATION
# =========================================================

@app.get("/webhook")
async def verify_webhook(request: Request):

    params = request.query_params

    mode = params.get("hub.mode")
    token = params.get("hub.verify_token")
    challenge = params.get("hub.challenge")

    if mode == "subscribe" and token == VERIFY_TOKEN:

        print("✅ Webhook verified successfully!")

        return PlainTextResponse(challenge)

    return PlainTextResponse(
        "Verification failed",
        status_code=403
    )


# =========================================================
# RECEIVE WHATSAPP MESSAGE
# =========================================================

@app.post("/webhook")
async def receive_message(request: Request):

    body = await request.json()

    print("\n==============================")
    print("Incoming WhatsApp data")
    print(body)
    print("==============================")

    try:

        entry = body["entry"][0]
        change = entry["changes"][0]
        value = change["value"]

        messages = value.get("messages")

        if not messages:

            return {"status": "no message"}

        message = messages[0]

        user_id = message["from"]
        message_type = message.get("type")

        # -------------------------------------------------
        # TEXT
        # -------------------------------------------------

        if message_type == "text":

            text = message["text"]["body"].strip()

            print("User:", user_id)
            print("Text:", text)

            state = user_states.get(user_id, "menu")

            # =============================================
            # MAIN MENU
            # =============================================

            if text.lower() in [
                "hi",
                "hello",
                "hey",
                "start",
                "menu"
            ]:

                user_states[user_id] = "menu"

                send_whatsapp_message(
                    user_id,
                    main_menu()
                )

                return {"status": "menu sent"}

            # =============================================
            # MAIN MENU OPTION 1
            # =============================================

            if state == "menu" and text == "1":

                user_states[user_id] = "identify"

                send_whatsapp_message(
                    user_id,
                    (
                        "🔍 *Identify an Unknown Dog*\n\n"
                        "Please send a clear nose photo "
                        "or a short 2-second nose video.\n\n"
                        "The AI identification pipeline "
                        "will process it."
                    )
                )

                return {"status": "identify selected"}

            # =============================================
            # MAIN MENU OPTION 2
            # =============================================

            if state == "menu" and text == "2":

                reply = start_registration(user_id)

                send_whatsapp_message(
                    user_id,
                    reply
                )

                return {"status": "registration started"}

            # =============================================
            # MAIN MENU OPTION 3
            # =============================================

            if state == "menu" and text == "3":

                user_states[user_id] = "update_dog_id"

                send_whatsapp_message(
                    user_id,
                    (
                        "📝 *Update Existing Dog Record*\n\n"
                        "Please enter the Dog ID.\n\n"
                        "Example:\n"
                        "DOG-4F0FB602"
                    )
                )

                return {"status": "update started"}

            # =============================================
            # REGISTRATION - NAME
            # =============================================

            if state == "registration_name":

                reply = handle_registration_name(
                    user_id,
                    text
                )

                send_whatsapp_message(
                    user_id,
                    reply
                )

                return {"status": "name received"}

            # =============================================
            # REGISTRATION - AREA
            # =============================================

            if state == "registration_area":

                reply = handle_registration_area(
                    user_id,
                    text
                )

                send_whatsapp_message(
                    user_id,
                    reply
                )

                return {"status": "area received"}

            # =============================================
            # REGISTRATION - MEDICAL
            # =============================================

            if state == "registration_medical":

                reply = handle_registration_medical(
                    user_id,
                    text
                )

                send_whatsapp_message(
                    user_id,
                    reply
                )

                return {"status": "medical received"}

            # =============================================
            # REGISTRATION CONFIRMATION
            # =============================================

            if state == "registration_confirm":

                if text == "1":

                    send_whatsapp_message(
                        user_id,
                        (
                            "✅ *Registration confirmed!*\n\n"
                            "The dog profile is ready "
                            "to be saved.\n\n"
                            "Dog ID and QR generation "
                            "will be connected to the "
                            "database/QR module."
                        )
                    )

                    user_states[user_id] = "menu"

                    return {
                        "status": "registration confirmed"
                    }

                elif text == "2":

                    user_states[user_id] = (
                        "registration_change_area"
                    )

                    send_whatsapp_message(
                        user_id,
                        (
                            "📍 Please enter the new "
                            "area/locality."
                        )
                    )

                    return {
                        "status": "change area"
                    }

                elif text == "3":

                    user_states[user_id] = (
                        "registration_change_medical"
                    )

                    send_whatsapp_message(
                        user_id,
                        (
                            "🏥 Please enter the new "
                            "medical details.\n\n"
                            "Example:\n"
                            "Vaccination: Yes\n"
                            "Sterilization: No"
                        )
                    )

                    return {
                        "status": "change medical"
                    }

                else:

                    send_whatsapp_message(
                        user_id,
                        (
                            "❌ Please reply with 1, 2 or 3.\n\n"
                            + confirmation_message(user_id)
                        )
                    )

                    return {
                        "status": "invalid confirmation"
                    }

            # =============================================
            # CHANGE AREA
            # =============================================

            if state == "registration_change_area":

                reply = change_area(
                    user_id,
                    text
                )

                send_whatsapp_message(
                    user_id,
                    reply
                )

                return {
                    "status": "area changed"
                }

            # =============================================
            # CHANGE MEDICAL
            # =============================================

            if state == "registration_change_medical":

                reply = change_medical(
                    user_id,
                    text
                )

                send_whatsapp_message(
                    user_id,
                    reply
                )

                return {
                    "status": "medical changed"
                }

            # =============================================
            # UPDATE EXISTING DOG - DOG ID
            # =============================================

            if state == "update_dog_id":

                dog_id = text.upper().strip()

                # TEMPORARY DEMO VERSION
                # Database team will replace this
                # with actual database lookup.

                if not dog_id.startswith("DOG-"):

                    send_whatsapp_message(
                        user_id,
                        (
                            "❌ Invalid Dog ID format.\n\n"
                            "Example:\n"
                            "DOG-4F0FB602"
                        )
                    )

                    return {
                        "status": "invalid dog id"
                    }

                user_update_dogs[user_id] = dog_id

                user_states[user_id] = "update_medical"

                send_whatsapp_message(
                    user_id,
                    (
                        "✅ Dog ID received.\n\n"
                        f"🆔 Dog ID: {dog_id}\n\n"
                        "Please enter the new medical details.\n\n"
                        "Example:\n"
                        "Vaccination: Yes\n"
                        "Sterilization: No"
                    )
                )

                return {
                    "status": "dog id received"
                }

            # =============================================
            # UPDATE MEDICAL
            # =============================================

            if state == "update_medical":

                medical = text.strip()

                if not medical:

                    send_whatsapp_message(
                        user_id,
                        "❌ Medical details cannot be empty."
                    )

                    return {
                        "status": "empty medical"
                    }

                if (
                    "vaccination" not in medical.lower()
                    and
                    "sterilization" not in medical.lower()
                ):

                    send_whatsapp_message(
                        user_id,
                        (
                            "❌ Please include vaccination "
                            "or sterilization details.\n\n"
                            "Example:\n"
                            "Vaccination: Yes\n"
                            "Sterilization: No"
                        )
                    )

                    return {
                        "status": "invalid medical"
                    }

                dog_id = user_update_dogs.get(user_id)

                send_whatsapp_message(
                    user_id,
                    (
                        "✅ *Medical details received!*\n\n"
                        f"🆔 Dog ID: {dog_id}\n"
                        f"🏥 Medical Details: {medical}\n\n"
                        "The database update will be "
                        "connected to the Database/Search "
                        "module."
                    )
                )

                del user_update_dogs[user_id]

                user_states[user_id] = "menu"

                return {
                    "status": "medical received"
                }

            # =============================================
            # DEFAULT TEXT
            # =============================================

            send_whatsapp_message(
                user_id,
                (
                    "❌ I didn't understand that.\n\n"
                    + main_menu()
                )
            )

        # -------------------------------------------------
        # IMAGE / VIDEO
        # -------------------------------------------------

        elif message_type in ["image", "video"]:

            state = user_states.get(
                user_id,
                "menu"
            )

            if state == "registration_nose":

                reply = handle_registration_nose(
                    user_id,
                    message_type
                )

                send_whatsapp_message(
                    user_id,
                    reply
                )

                return {
                    "status": "nose media received"
                }

            if state == "identify":

                send_whatsapp_message(
                    user_id,
                    (
                        "📷 Nose media received.\n\n"
                        "This will be passed to the "
                        "nose-print recognition pipeline."
                    )
                )

                return {
                    "status": "identify media received"
                }

            send_whatsapp_message(
                user_id,
                (
                    "❌ Please use the menu to select "
                    "an option first.\n\n"
                    + main_menu()
                )
            )

    except Exception as e:

        print("ERROR:", e)

    return {"status": "received"}
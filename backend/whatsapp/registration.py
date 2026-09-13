from state import user_states, user_registration


def start_registration(user_id):

    user_states[user_id] = "registration_nose"

    user_registration[user_id] = {}

    return (
        "🐶 *New Dog Registration*\n\n"
        "Please send a clear nose photo "
        "or a short 2-second nose video."
    )


def handle_registration_nose(user_id, message_type):

    if message_type not in ["image", "video"]:

        return (
            "❌ Please send a nose photo or a "
            "short 2-second nose video.\n\n"
            "This is required to create the dog's "
            "nose-print identity."
        )

    # For now we only record that nose media was received.
    # Later this will connect to the AI team's pipeline.

    user_registration[user_id]["nose_received"] = True

    user_states[user_id] = "registration_name"

    return (
        "✅ Nose image/video received.\n\n"
        "What is the dog's name or nickname?\n\n"
        "Example: Bruno"
    )


def handle_registration_name(user_id, text):

    name = text.strip()

    if not name:

        name = "Unnamed"

    user_registration[user_id]["name"] = name

    user_states[user_id] = "registration_area"

    return (
        "📍 What area or locality is the dog from?\n\n"
        "Example: Indiranagar"
    )


def handle_registration_area(user_id, text):

    area = text.strip()

    if not area:

        return (
            "❌ Area cannot be empty.\n\n"
            "Please enter the area/locality."
        )

    if area.isdigit():

        return (
            "❌ Please enter a valid area/location."
        )

    user_registration[user_id]["area"] = area

    user_states[user_id] = "registration_medical"

    return (
        "🏥 Please enter the medical details.\n\n"
        "Example:\n"
        "Vaccination: Yes\n"
        "Sterilization: No"
    )


def handle_registration_medical(user_id, text):

    medical = text.strip()

    if not medical:

        return (
            "❌ Medical details cannot be empty.\n\n"
            "Example:\n"
            "Vaccination: Yes\n"
            "Sterilization: No"
        )

    lower = medical.lower()

    if (
        "vaccination" not in lower
        and "sterilization" not in lower
    ):

        return (
            "❌ Please include vaccination "
            "or sterilization details.\n\n"
            "Example:\n"
            "Vaccination: Yes\n"
            "Sterilization: No"
        )

    user_registration[user_id]["medical_details"] = medical

    user_states[user_id] = "registration_confirm"

    return confirmation_message(user_id)


def confirmation_message(user_id):

    data = user_registration[user_id]

    return (
        "📋 *Please confirm the registration details:*\n\n"
        f"🐶 Name: {data.get('name', 'Unnamed')}\n"
        f"📍 Area: {data.get('area', '')}\n"
        f"🏥 Medical Details: "
        f"{data.get('medical_details', '')}\n\n"
        "Reply with:\n\n"
        "1️⃣ Confirm\n"
        "2️⃣ Change Area\n"
        "3️⃣ Change Medical Details"
    )

def change_area(user_id, text):

    new_area = text.strip()

    if not new_area:

        return (
            "❌ Area cannot be empty.\n\n"
            "Please enter the area."
        )

    if new_area.isdigit():

        return (
            "❌ Please enter a valid area/location."
        )

    user_registration[user_id]["area"] = new_area

    user_states[user_id] = "registration_confirm"

    return confirmation_message(user_id)


def change_medical(user_id, text):

    new_medical = text.strip()

    if not new_medical:

        return (
            "❌ Medical details cannot be empty."
        )

    lower = new_medical.lower()

    if (
        "vaccination" not in lower
        and "sterilization" not in lower
    ):

        return (
            "❌ Please include vaccination "
            "or sterilization details.\n\n"
            "Example:\n"
            "Vaccination: Yes\n"
            "Sterilization: No"
        )

    user_registration[user_id]["medical_details"] = new_medical

    user_states[user_id] = "registration_confirm"

    return confirmation_message(user_id)
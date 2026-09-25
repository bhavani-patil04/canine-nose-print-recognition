# Living Bridges - WhatsApp Integration

This module provides the WhatsApp integration prototype for the
Living Bridges Canine Nose-Print Recognition project.

## Features

- WhatsApp Cloud API webhook
- WhatsApp message receiving
- Automated WhatsApp replies
- 3-option main menu
- Unknown dog identification flow
- Known dog registration flow
- Dog name validation
- Area/location validation
- Medical details validation
- Registration confirmation
- Dog ID generation
- QR code generation
- Existing dog record update
- SQLite database
- Local testing endpoint
- Browser-based WhatsApp-style testing UI

## Project Structure

```text
whatsapp integration/
│
├── main.py
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
│
├── dogs.db              # Created automatically
└── qr_codes/            # Created automatically
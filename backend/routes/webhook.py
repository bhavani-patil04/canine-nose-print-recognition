from fastapi import APIRouter, Request

router = APIRouter()


@router.get("/webhook")
def verify_webhook():
    return {
        "status": "success",
        "message": "Webhook verification endpoint is ready"
    }


@router.post("/webhook")
async def receive_webhook(request: Request):
    data = await request.json()

    return {
        "status": "success",
        "message": "Webhook received",
        "data": data
    }
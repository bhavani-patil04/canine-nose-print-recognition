from fastapi import FastAPI
from backend.routes.health import router as health_router
from backend.routes.identify import router as identify_router

app = FastAPI(
    title="Canine Nose-Print Recognition API",
    description="Backend API for the Canine Nose-Print Recognition Prototype",
    version="1.0.0"
)

app.include_router(health_router)
app.include_router(identify_router)
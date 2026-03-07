from fastapi import FastAPI
from app.api.scan import router as scan_router

app = FastAPI(
    title="SafeQR AI Security API",
    description="AI-powered QR code phishing detection system",
    version="1.0"
)

app.include_router(scan_router, prefix="/api")

@app.get("/")
def root():
    return {"message": "SafeQR Backend Running"}
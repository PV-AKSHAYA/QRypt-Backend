"""
app/main.py
───────────
FastAPI application entry point.
- Creates the app instance
- Registers middleware (CORS, rate limiting)
- Mounts all routers
- Health check endpoint
- Startup / shutdown lifecycle hooks
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
import logging

from app.core.config import settings

# ── Logger ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger("safeqr")


# ── Lifespan (startup / shutdown) ────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ──────────────────────────────────────────────
    logger.info(f"🚀 {settings.APP_NAME} v{settings.APP_VERSION} starting...")

    # Connect MongoDB
    try:
        from app.database.db import connect_db, disconnect_db
        await connect_db()
        logger.info("✅ MongoDB connected")
    except Exception as e:
        logger.warning(f"⚠️  MongoDB not available: {e} — scans will still work")

    logger.info("✅ All services ready")
    yield

    # ── Shutdown ─────────────────────────────────────────────
    try:
        await disconnect_db()
        logger.info("MongoDB disconnected")
    except Exception:
        pass
    logger.info("SafeQR shutdown complete")


# ── App Instance ─────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Multi-modal forensic QR code security scanner",
    docs_url="/docs",          # Swagger UI at /docs
    redoc_url="/redoc",
    lifespan=lifespan,
)


# ── CORS Middleware ───────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # tighten to your domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Timing Middleware ─────────────────────────────────
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)
    response.headers["X-Process-Time-Ms"] = str(duration)
    logger.debug(f"{request.method} {request.url.path} — {duration}ms")
    return response


# ── Global Exception Handler ──────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled error on {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else "Contact support",
        },
    )


# ── Health Check ─────────────────────────────────────────────
@app.get("/health", tags=["System"])
async def health_check():
    """
    Liveness probe.
    Render.com uses this to verify the service is running.
    """
    return {
        "status": "ok",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "layers": [
            "physical_analyzer",
            "redirect_engine",
            "threat_intel",
            "ai_context_engine",
            "risk_engine",
        ],
    }


# ── Routers ───────────────────────────────────────────────────
from app.api.scan import router as scan_router
app.include_router(scan_router, prefix="/api/v1", tags=["Scan"])


# ── Dev entrypoint ────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="debug" if settings.DEBUG else "info",
    )
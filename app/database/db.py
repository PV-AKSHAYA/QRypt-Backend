"""
app/database/db.py
───────────────────
MongoDB connection manager using Motor (async driver).

Design:
  - connect_db() called at startup
  - disconnect_db() called at shutdown
  - get_db() returns the database instance anywhere in the app
  - If MONGODB_URI is empty, all DB ops silently skip (scan still works)
"""

import logging
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase

logger = logging.getLogger("safeqr.db")

_client: AsyncIOMotorClient | None   = None
_db:     AsyncIOMotorDatabase | None = None


async def connect_db() -> None:
    """
    Open MongoDB connection at app startup.
    Raises if URI is set but connection fails.
    Silently skips if URI is empty.
    """
    global _client, _db
    from app.core.config import settings

    if not settings.MONGODB_URI:
        raise ConnectionError("MONGODB_URI not configured")

    _client = AsyncIOMotorClient(
        settings.MONGODB_URI,
        serverSelectionTimeoutMS = 5000,
        connectTimeoutMS         = 5000,
    )

    # Ping to verify connection is alive
    await _client.admin.command("ping")

    _db = _client[settings.MONGODB_DB_NAME]

    # Ensure indexes exist for fast lookups
    await _db["scans"].create_index("final_domain")
    await _db["scans"].create_index("image_hash")
    await _db["scans"].create_index("timestamp")

    logger.info(f"MongoDB connected: db={settings.MONGODB_DB_NAME}")


async def disconnect_db() -> None:
    """Close MongoDB connection at app shutdown."""
    global _client, _db
    if _client:
        _client.close()
        _client = None
        _db     = None
        logger.info("MongoDB disconnected")


def get_db() -> AsyncIOMotorDatabase:
    """
    Return the database instance.
    Raises RuntimeError if not connected (caught by callers).
    """
    if _db is None:
        raise RuntimeError("MongoDB not connected")
    return _db
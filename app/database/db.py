"""
app/database/db.py
───────────────────
MongoDB async client + Threat Memory Engine.

Two public interfaces:
  1. connect_db() / disconnect_db()    — called in main.py lifespan
  2. ThreatMemoryEngine                — lookup + save scan history
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from app.core.config import settings

logger = logging.getLogger("safeqr.db")

_client = None
_db     = None


# ══════════════════════════════════════════════════════════════
#  CONNECTION
# ══════════════════════════════════════════════════════════════

async def connect_db():
    global _client, _db
    import motor.motor_asyncio
    _client = motor.motor_asyncio.AsyncIOMotorClient(settings.MONGODB_URI)
    _db     = _client[settings.MONGODB_DB_NAME]

    # Create indexes
    await _db.scans.create_index("scan_id",     unique=True)
    await _db.scans.create_index("final_domain")
    await _db.scans.create_index("image_hash")
    await _db.scans.create_index("timestamp")
    await _db.threat_memory.create_index("domain", unique=True)

    logger.info(f"MongoDB connected — db={settings.MONGODB_DB_NAME}")


async def disconnect_db():
    global _client
    if _client:
        _client.close()
        logger.info("MongoDB disconnected")


def get_db():
    if _db is None:
        raise RuntimeError("Database not connected")
    return _db


# ══════════════════════════════════════════════════════════════
#  THREAT MEMORY ENGINE
# ══════════════════════════════════════════════════════════════

class ThreatMemoryEngine:
    """
    Looks up and updates domain reputation across scans.

    Usage in scan.py:
        memory = await ThreatMemoryEngine.lookup(domain)
        # ... run scan ...
        await ThreatMemoryEngine.record(domain, verdict, vt_malicious)
    """

    @staticmethod
    async def lookup(domain: str) -> dict:
        """
        Check if this domain has been seen before.
        Returns threat memory summary for the scan response.
        """
        try:
            db  = get_db()
            doc = await db.threat_memory.find_one({"domain": domain})

            if not doc:
                return {
                    "seen_before":         False,
                    "previous_scan_count": 0,
                    "first_seen":          None,
                    "last_verdict":        None,
                    "times_flagged":       0,
                    "flagged":             False,
                }

            return {
                "seen_before":         True,
                "previous_scan_count": doc.get("scan_count", 0),
                "first_seen":          doc.get("first_seen", "").isoformat()
                                       if isinstance(doc.get("first_seen"), datetime)
                                       else doc.get("first_seen"),
                "last_verdict":        doc.get("last_verdict", "UNKNOWN"),
                "times_flagged":       doc.get("high_risk_count", 0),
                "flagged":             doc.get("flagged", False),
            }

        except Exception as e:
            logger.debug(f"Threat memory lookup skipped: {e}")
            return {
                "seen_before":         False,
                "previous_scan_count": 0,
                "first_seen":          None,
                "last_verdict":        None,
                "times_flagged":       0,
                "flagged":             False,
            }

    @staticmethod
    async def record(domain: str, verdict: str, vt_malicious: int = 0):
        """
        Upsert domain reputation after a scan completes.
        Creates entry on first seen, increments counters thereafter.
        """
        try:
            db  = get_db()
            now = datetime.now(timezone.utc)

            verdict_upper = verdict.upper()
            inc = {
                "scan_count":         1,
                "total_vt_malicious": vt_malicious,
            }

            if verdict_upper == "HIGH_RISK":
                inc["high_risk_count"] = 1
            elif verdict_upper == "SUSPICIOUS":
                inc["suspicious_count"] = 1
            else:
                inc["safe_count"] = 1

            await db.threat_memory.update_one(
                {"domain": domain},
                {
                    "$inc": inc,
                    "$set": {
                        "last_seen":    now,
                        "last_verdict": verdict_upper,
                        "flagged":      verdict_upper == "HIGH_RISK",
                    },
                    "$setOnInsert": {
                        "first_seen": now,
                    },
                },
                upsert=True,
            )
            logger.debug(f"Threat memory updated: domain={domain} verdict={verdict_upper}")

        except Exception as e:
            logger.debug(f"Threat memory record skipped: {e}")

    @staticmethod
    async def check_duplicate_image(image_hash: str) -> Optional[dict]:
        """
        Check if this exact image was scanned before (by MD5 hash).
        Returns previous scan result or None.
        """
        try:
            db  = get_db()
            doc = await db.scans.find_one(
                {"image_hash": image_hash},
                sort=[("timestamp", -1)]    # most recent first
            )
            if doc:
                return {
                    "duplicate":      True,
                    "previous_scan":  doc.get("scan_id"),
                    "previous_verdict": doc.get("verdict"),
                    "scanned_at":     doc.get("timestamp", "").isoformat()
                                      if isinstance(doc.get("timestamp"), datetime)
                                      else doc.get("timestamp"),
                }
            return None

        except Exception as e:
            logger.debug(f"Duplicate check skipped: {e}")
            return None

    @staticmethod
    async def save_scan(scan_doc: dict):
        """
        Save full scan document to the scans collection.
        """
        try:
            db = get_db()
            await db.scans.insert_one(scan_doc)
            logger.debug(f"Scan saved: scan_id={scan_doc.get('scan_id')}")
        except Exception as e:
            logger.debug(f"DB save skipped: {e}")
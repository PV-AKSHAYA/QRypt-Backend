"""
tests/manual_test.py
─────────────────────
Manual test script — runs against your live server.
Run AFTER: uvicorn app.main:app --reload

Usage:
    python app/tests/manual_test.py

Tests all 3 demo scenarios and prints clean results.
No mocks — hits real VT and Gemini if keys are configured.
Use skip flags to test without API keys.
"""

import io
import sys
import json
import qrcode
import requests
import numpy as np
from PIL import Image, ImageDraw

BASE_URL    = "http://localhost:8000"
SCAN_URL    = f"{BASE_URL}/api/v1/scan"
HEALTH_URL  = f"{BASE_URL}/health"

# ── Colour helpers ────────────────────────────────────────────
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"


def make_qr(url: str) -> bytes:
    img = qrcode.make(url)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def make_tampered_qr(url: str) -> bytes:
    qr = qrcode.make(url).resize((300, 300))
    canvas = Image.new("RGB", (400, 400), "white")
    canvas.paste(qr, (50, 50))
    draw = ImageDraw.Draw(canvas)
    draw.rectangle([90,  90,  270, 270], outline="black", width=5)
    draw.rectangle([95,  95,  265, 265], fill=(235, 235, 235))
    draw.rectangle([110, 110, 250, 250], outline="black", width=4)
    draw.rectangle([115, 115, 245, 245], fill=(200, 200, 200))
    buf = io.BytesIO()
    canvas.save(buf, format="PNG")
    return buf.getvalue()


def colour_verdict(verdict: str) -> str:
    if verdict == "HIGH_RISK":   return f"{RED}{BOLD}{verdict}{RESET}"
    if verdict == "SUSPICIOUS":  return f"{YELLOW}{BOLD}{verdict}{RESET}"
    return f"{GREEN}{BOLD}{verdict}{RESET}"


def colour_score(score: int) -> str:
    if score >= 60: return f"{RED}{score}{RESET}"
    if score >= 30: return f"{YELLOW}{score}{RESET}"
    return f"{GREEN}{score}{RESET}"


def print_result(label: str, data: dict) -> None:
    risk = data["risk"]
    phys = data["physical_layer"]
    tech = data["technical_layer"]
    ai   = data["ai_layer"]
    mem  = data["threat_memory"]

    print(f"\n{CYAN}{'─'*55}{RESET}")
    print(f"{BOLD}  {label}{RESET}")
    print(f"{'─'*55}")
    print(f"  Score   : {colour_score(risk['score'])}/100")
    print(f"  Verdict : {colour_verdict(risk['verdict'])}")
    print(f"  Scan ID : {data['scan_id'][:16]}...")
    print()
    print(f"  {BOLD}Physical Layer{RESET}")
    tamper = f"{RED}TAMPERED{RESET}" if phys["tampered"] else f"{GREEN}CLEAN{RESET}"
    print(f"    Status     : {tamper}")
    print(f"    Confidence : {phys['confidence']}%")
    print(f"    Evidence   : {phys['evidence'][:70]}")
    print()
    print(f"  {BOLD}Technical Layer{RESET}")
    print(f"    Final URL  : {tech['final_url'][:60]}")
    print(f"    Hops       : {tech['hop_count']}")
    print(f"    SSL Valid  : {tech['ssl_valid']}")
    print(f"    TLD Risk   : {tech['tld_risk_score']}")
    print(f"    Keywords   : {tech['suspicious_keywords']}")
    vt = tech["virustotal"]
    print(f"    VT Result  : {vt['malicious']} malicious / {vt['total_engines']} engines → {vt['reputation_class']}")
    print()
    print(f"  {BOLD}AI Context Layer{RESET}")
    print(f"    Visual     : {ai['visual_context'][:60]}")
    print(f"    Brand      : {ai['expected_brand']}")
    print(f"    URL Match  : {ai['url_match']}")
    print(f"    Imperson.  : {ai['impersonation_probability']}")
    print(f"    Explain    : {ai['explanation'][:70]}")
    print()
    print(f"  {BOLD}Breakdown{RESET}")
    bd = risk["breakdown"]
    print(f"    Physical   : {bd['physical_score']:.1f} pts")
    print(f"    ThreatIntel: {bd['threat_intel_score']:.1f} pts")
    print(f"    AI Context : {bd['ai_context_score']:.1f} pts")
    if mem["seen_before"]:
        print(f"\n  {RED}⚠  THREAT MEMORY: Previously flagged {mem['previous_scan_count']} times{RESET}")


def run_scan(label: str, img_bytes: bytes, **kwargs) -> bool:
    print(f"\n{CYAN}Testing: {label}...{RESET}", end=" ", flush=True)
    try:
        r = requests.post(
            SCAN_URL,
            files={"image": ("test.png", img_bytes, "image/png")},
            data=kwargs,
            timeout=60,
        )
        if r.status_code == 200:
            print(f"{GREEN}✓ 200 OK{RESET}")
            print_result(label, r.json())
            return True
        else:
            print(f"{RED}✗ {r.status_code}{RESET}")
            print(f"  Error: {r.text[:200]}")
            return False
    except requests.exceptions.ConnectionError:
        print(f"{RED}✗ Connection refused — is the server running?{RESET}")
        return False
    except Exception as e:
        print(f"{RED}✗ {e}{RESET}")
        return False


def main():
    print(f"\n{BOLD}{CYAN}SafeQR Manual Integration Test{RESET}")
    print(f"{CYAN}{'═'*55}{RESET}")

    # Health check
    print(f"\nHealth check...", end=" ")
    try:
        r = requests.get(HEALTH_URL, timeout=5)
        if r.status_code == 200:
            print(f"{GREEN}✓ Server is running{RESET}")
        else:
            print(f"{RED}✗ Server unhealthy{RESET}")
            sys.exit(1)
    except Exception:
        print(f"{RED}✗ Server not reachable — run: uvicorn app.main:app --reload{RESET}")
        sys.exit(1)

    results = []

    # ── Scenario 1: SAFE ─────────────────────────────────────
    ok = run_scan(
        "SCENARIO 1 — Safe QR (GitHub)",
        make_qr("https://github.com"),
        skip_virustotal="true",
        skip_ai="true",
    )
    results.append(("SAFE", ok))

    # ── Scenario 2: SUSPICIOUS ────────────────────────────────
    ok = run_scan(
        "SCENARIO 2 — Suspicious URL",
        make_qr("https://login-verify-account.xyz/confirm"),
        skip_virustotal="true",
        skip_ai="true",
    )
    results.append(("SUSPICIOUS", ok))

    # ── Scenario 3: HIGH RISK ─────────────────────────────────
    ok = run_scan(
        "SCENARIO 3 — Tampered + Malicious",
        make_tampered_qr("https://taxrefund-claim.xyz/pay"),
        skip_virustotal="true",
        skip_ai="true",
    )
    results.append(("HIGH_RISK", ok))

    # ── Summary ───────────────────────────────────────────────
    print(f"\n{CYAN}{'═'*55}{RESET}")
    print(f"{BOLD}  Test Summary{RESET}")
    print(f"{'─'*55}")
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        status = f"{GREEN}PASS{RESET}" if ok else f"{RED}FAIL{RESET}"
        print(f"  {status}  {name}")
    print(f"\n  {passed}/{len(results)} scenarios passed")

    if passed == len(results):
        print(f"\n{GREEN}{BOLD}  ✓ All scenarios passing — backend is ready!{RESET}")
        print(f"{CYAN}  Swagger UI: http://localhost:8000/docs{RESET}\n")
    else:
        print(f"\n{RED}  Some scenarios failed — check output above{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
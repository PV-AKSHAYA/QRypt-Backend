# QRypt-Backend
This repo consists source code of backend


# SafeQR Backend

SafeQR is a cybersecurity-focused backend service that analyzes QR codes to detect potentially malicious URLs.
It extracts QR data from uploaded images, analyzes the embedded URL, follows redirect chains, evaluates risk signals, and returns a security verdict.

The backend is built using **FastAPI** and provides APIs for QR code scanning and threat analysis.

---

## Project Architecture

```
safeqr-backend
│
├── app
│   ├── api
│   │   └── scan.py
│   │
│   ├── core
│   │   ├── config.py
│   │   └── security.py
│   │
│   ├── models
│   │   ├── request_models.py
│   │   └── response_models.py
│   │
│   ├── services
│   │   ├── qr_extractor.py
│   │   ├── redirect_engine.py
│   │   ├── threat_intel.py
│   │   ├── ai_context_engine.py
│   │   └── risk_engine.py
│   │
│   ├── utils
│   │   ├── image_utils.py
│   │   ├── url_utils.py
│   │   └── validators.py
│   │
│   └── main.py
│
├── tests
├── requirements.txt
└── README.md
```

---

# Features

* QR Code Extraction from uploaded images
* URL validation and domain analysis
* Redirect chain detection
* Suspicious domain pattern detection
* Threat intelligence checks
* Risk scoring engine
* REST API for integration with frontend applications

---

# Tech Stack

* Python
* FastAPI
* Uvicorn
* OpenCV / Pillow
* tldextract
* validators

---

# Setup Instructions

Follow these steps to run the backend locally.

## 1. Clone the Repository

```
git clone <MAIN_REPOSITORY_URL>
cd safeqr-backend
```

Example:

```
git clone https://github.com/ORG_NAME/QRypt-Backend.git
cd QRypt-Backend
```

---

## 2. Create a Virtual Environment

Create a Python virtual environment.

```
python -m venv venv
```

---

## 3. Activate Virtual Environment

### Windows

```
venv\Scripts\activate
```

### Mac/Linux

```
source venv/bin/activate
```

---

## 4. Install Dependencies

Install all required packages.

```
pip install -r requirements.txt
```

---

## 5. Run the FastAPI Server

Start the backend using Uvicorn.

```
uvicorn app.main:app --reload
```

Server will start at:

```
http://127.0.0.1:8000
```

---

# API Documentation

FastAPI automatically generates interactive documentation.

Open in your browser:

```
http://127.0.0.1:8000/docs
```

This provides the Swagger interface where you can test the APIs.

---

# Example API Endpoint

## Scan QR Code

Endpoint:

```
POST /scan
```

Upload a QR image file and the backend will return the security analysis.

Example Response:

```
{
  "status": "success",
  "qr_url": "https://example.com/login",
  "redirect_chain": [],
  "risk_score": 32,
  "verdict": "Suspicious"
}
```

---

# Development Notes

* Do **not commit the virtual environment folder (`venv/`)**
* Use `.gitignore` to exclude sensitive files such as `.env`
* Always regenerate `requirements.txt` if new dependencies are added

```
pip freeze > requirements.txt
```

---

# Future Improvements

* Threat intelligence API integrations
* Phishing detection using AI
* QR threat map visualization
* Browser extension for real-time scanning
* Mobile QR scanning support

---

# Contributors

Backend implementation contributed through pull requests to the main repository.

---

# License

This project is intended for research and educational use in cybersecurity and secure QR code analysis.

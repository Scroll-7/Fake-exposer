# Fake Exposer

Multi-modal fake news / misinformation detection platform. Analyzes text, URLs, and images using Groq LLM, Gemini Vision AI, and Python ML models.

## Quick Start

```bash
npm install
pip install -r requirements.txt
npm start
```

Open `http://localhost:3001`.

## Architecture

| Service | Port | Technology |
|---------|------|------------|
| Frontend + API | 3001 | Node.js / Express |
| Text ML | 8000 | Python / FastAPI + scikit-learn |
| Image ML | 8001 | Python / FastAPI + ONNX Runtime |
| Face ID | 8002 | Python / FastAPI + InsightFace |

The Node.js server auto-spawns all 3 Python APIs as child processes on startup.

## Scripts

- `npm start` — launch server (auto-spawns all Python APIs)
- `npm test` — run all tests (unit + integration)
- `npm run test:unit` — unit tests only
- `python auto_train.py` — (re)train all ML models
- `start.bat` — wrapper with prereq checks and auto-install

## Environment

Copy `.env.example` to `.env` and configure:

- `GROQ_API_KEY` — Groq LLM for text/image analysis
- `GEMINI_API_KEYS` — Google Gemini for AI forensics + sports ID
- `PORT` — server port (default 3001)

## Tests

20 unit tests + 4 integration tests (skip when server offline). Run with `npm test`.

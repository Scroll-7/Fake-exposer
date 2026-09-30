# Fake News Detector

A multi-modal misinformation and deepfake detection platform. Give it **text**, a **URL**, or an **image**, and it returns a credibility score (0–100) with an evidence-backed verdict.

It combines frontier LLMs (Groq LLaMA 4 vision, Google Gemini), three local Python ML microservices (InsightFace, ONNX, scikit-learn), and a set of deterministic rule-based forensic heuristics — then reconciles them with hard score caps so a suspicious signal can never be talked away by a generous LLM.

---

## Table of Contents

- [Why this exists](#why-this-exists)
- [Features](#features)
- [How the analysis actually works](#how-the-analysis-actually-works)
- [Architecture](#architecture)
- [Tech stack](#tech-stack)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [API reference](#api-reference)
- [Project structure](#project-structure)
- [Security](#security)
- [Testing](#testing)
- [Extending](#extending)
- [Troubleshooting](#troubleshooting)
- [License](#license)

---

## Why this exists

Generative image models produce "AI muscle enhancement" photos that look convincing at a glance: a fit person in a mirror selfie with implausibly sculpted arms, or a studio-perfect portrait with flawless plastic skin. These get reposted as "proof," and general-purpose chatbots often answer "this looks fine."

This project attacks that specific failure mode with three things a single LLM call cannot do:

1. **Ground truth, not vibes.** InsightFace does a mathematical face match against known athletes, so the system knows *who* is actually in the photo instead of guessing from pixels.
2. **Knowledge-base fact-checking.** A local player↔club database catches the classic face-swap fake: a real photo of a player with the jersey digitally swapped to a team they never played for.
3. **Physics and forensics heuristics.** Muscle-to-skeleton ratio, skin-texture mismatch between face and torso, lighting geometry across body parts, and edge artifacts at the body silhouette — the exact fingerprints left by AI muscle editors.

---

## Features

| Capability | What it does |
|---|---|
| **Image analysis** | Detects AI generation, muscle enhancement, deepfakes, jersey swaps, and screenshot tampering. Accepts JPEG/PNG up to 4 MB. |
| **URL analysis** | Scrapes any public article and fact-checks it with live web-search context. |
| **Text analysis** | Full Groq LLM fact-check pipeline with source-reputation scoring. |
| **ZeroGPT-style text detector** | 7-metric local statistical AI-text detector. **No API key, no network, no cost.** |
| **Face ID** | InsightFace recognition of 32 known public figures, wired into a jersey-mismatch knowledge base. |
| **Celebrity context** | Known roles, organizations, and typical settings for each public figure, to flag out-of-context imagery. |
| **Local AI-image classifier** | ONNX model scores image-level AI probability, merged as an override on the final score. |
| **Trusted-source boost** | Verified/prominent sources (Reuters, BBC, Sky Sports, …) receive a documented +50 credibility boost. |
| **Multi-provider fallback** | 3 Groq vision models + 4 Gemini models with API-key rotation, so one provider being down or rate-limited doesn't break analysis. |

---

## How the analysis actually works

Image analysis is a **deterministic pipeline**, not an agent loop. Every stage runs exactly once, in a fixed order — which keeps latency, cost, and failure behavior predictable.

```
                    ┌─────────────────────────────────────────┐
   uploaded  ──────▶│  STAGE 1 — Parallel probes (Promise.all)│
   image            │  Face ID :8002 │ Gemini │ ONNX :8001 │ Groq vision │
                    └──────────────────────┬──────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────┐
                    │  STAGE 2 — Local deterministic rules     │
                    │  jersey mismatch · unverifiable identity│
                    │  physique · portrait · filename          │
                    └──────────────────────┬──────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────┐
                    │  STAGE 3 — Groq LLaMA 4 vision (JSON)    │
                    │  final verdict with all context injected │
                    └──────────────────────┬──────────────────┘
                                           │
                    ┌──────────────────────▼──────────────────┐
                    │  STAGE 4 — Hard overrides & caps         │
                    │  Gemini AI verdict → jersey (8) →        │
                    │  physique (20) → filename (5) →          │
                    │  unverifiable identity (15) → sports (55)│
                    └──────────────────────┬──────────────────┘
                                           ▼
                              credibility_score 0–100 + verdict
```

**Stage 4 is the important part.** LLMs are generous by default — they tend to say "looks authentic" when uncertain. So the local detectors get the *last word*: a confirmed jersey mismatch hard-caps the score at 8 regardless of what the LLM claimed, and a filename revealing an AI tool pins it at 5. This is why the system is auditable: every cap is a rule in code you can read, not a vibe.

### Score caps (precedence order)

| Condition | Capped score |
|---|---|
| Gemini classifies as AI-generated | Gemini's score (hard override) |
| Player in a jersey they never played for | **8** |
| Extreme physique in a casual setting (AI muscle edit) | **20** |
| AI portrait signature (uniform bg + studio light) | proportional penalty |
| Filename reveals an AI tool | **5** |
| Team jersey but unidentifiable person | **15** |
| Sports image, no identifiable team/player | **55** |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│  Browser — public/ (vanilla HTML/CSS/JS, no build step)              │
└───────────────────────────────┬──────────────────────────────────────┘
                                │  fetch (JSON / multipart)
┌───────────────────────────────▼──────────────────────────────────────┐
│  server.js — Express                                                │
│  helmet CSP · CORS · hpp · gzip · rate limits · multer               │
│  magic-byte validation · SSRF scheme guard · trusted-source boost    │
└───┬───────────────┬───────────────┬───────────────┬──────────────────┘
    │               │               │               │
    ▼               ▼               ▼               ▼
groq.js         scraper.js     heuristics.js   textDetector.js
(orchestrator)   (Jina/DDG)     (pure rules)    (7-metric local)
    │                               │
    │  ┌────────────────────────────┼──────────────────────────┐
    ▼  ▼                            ▼                          ▼
Groq LLaMA 4    Gemini       sportsKB.js                  no network
vision + text   image forensics (player↔club KB)
    │
    ├──▶ http://127.0.0.1:8000/predict          (python_api.py — TF-IDF + LogReg)
    ├──▶ http://127.0.0.1:8001/detect_ai        (animal_api.py — ONNX AI detector)
    ├──▶ http://127.0.0.1:8001/predict_animal   (animal_api.py — image classifier)
    └──▶ http://127.0.0.1:8002/recognize        (face_api.py  — InsightFace)

    ┌──────────────────────────────────────────────────────────┐
    │ All three Python services are OPTIONAL. If one is down,  │
    │ the server logs a warning and degrades — it never dies.  │
    └──────────────────────────────────────────────────────────┘
```

---

## Tech stack

**Backend** — Node.js 24 · Express 4 · ES modules
**LLMs** — Groq (`groq-sdk`, LLaMA 4 Scout/Maverick + Qwen vision) · Google Gemini
**ML services** — Python · FastAPI · Uvicorn · scikit-learn · ONNX Runtime · InsightFace (buffalo_l)
**Frontend** — vanilla HTML/CSS/JS, no framework, no bundler
**Security** — helmet (CSP) · CORS · express-rate-limit · hpp · multer
**Testing** — `node:test` (99 tests) · ESLint 10 · Playwright (E2E)
**CI** — GitHub Actions

**Dependencies are deliberately minimal** (11 production packages). There is no LangChain, no agent framework, and no vector database: the orchestration is ~2 direct LLM calls, and a framework would add weight while hiding the provider-specific behavior (raw status codes, per-model fallback) this system depends on.

---

## Quick start

### Prerequisites

- **Node.js 18+** (CI runs Node 24)
- **Python 3.8+** — optional, for the local ML services
- A **Groq API key** — <https://console.groq.com/keys>
- A **Gemini API key** — <https://aistudio.google.com/apikey> (optional but recommended)

### 1. Install

```bash
git clone https://github.com/Scroll-7/Fake-exposer.git
cd Fake-exposer
npm install
```

### 2. Configure

```bash
cp .env.example .env
# edit .env and paste your keys
```

### 3. Run

```bash
npm start
```

Open **<http://localhost:3001>**.

> On Windows you can double-click **`start.bat`**, which checks prerequisites, installs npm + Python dependencies, and launches the server.

### 4. (Optional) Start the local ML services

The server **degrades gracefully** without these — every feature that depends on one is simply skipped. To enable full capability, run each in its own terminal:

```bash
pip install -r requirements.txt

python python_api.py     # :8000  ML text classification (TF-IDF + Logistic Regression)
python animal_api.py     # :8001  ONNX AI-image detection + image classification
python face_api.py       # :8002  InsightFace face recognition
```

The ONNX model (~83 MB) is not committed — download it once:

```bash
npm run download-models
```

---

## Configuration

All configuration is environment-based (see `.env.example`):

| Variable | Required | Default | Purpose |
|---|---|---|---|
| `GROQ_API_KEY` | ✅ | — | Text + vision analysis via LLaMA 4 |
| `GEMINI_API_KEYS` | ⭕ | — | AI image forensics. Comma-separated for rotation |
| `PORT` | ⭕ | `3001` | HTTP port |

Without `GROQ_API_KEY` the server still boots and serves the UI, but analysis endpoints return a descriptive error instead of crashing.

---

## API reference

Base URL: `http://localhost:3001`

### `POST /api/analyze/image`

Analyze an uploaded image. `multipart/form-data`, field name `image`. Optional text field `context` (e.g. `"is this AI generated?"`).

```bash
curl -X POST http://localhost:3001/api/analyze/image \
  -F "image=@photo.jpg" \
  -F "context=Is Neymar in a Real Madrid jersey real?"
```

```json
{
  "credibility_score": 8,
  "verdict": "Fake / Manipulated — Neymar never played for Real Madrid",
  "bias": "None",
  "sentiment": "Neutral",
  "red_flags": ["⚽ Jersey mismatch: Neymar has NEVER played for Real Madrid"],
  "green_flags": [],
  "summary": "...",
  "recommendations": ["..."],
  "face_identified": "neymar"
}
```

**Constraints:** JPEG/PNG only, ≤ 5 MB upload (≤ 4 MB base64 to the provider), 15 requests/hour/IP.

> GIF and WebP are rejected with a `400`. The vision providers cannot decode them, which previously caused silent "unable to analyze" results.

### `POST /api/analyze/text`

```bash
curl -X POST http://localhost:3001/api/analyze/text \
  -H "Content-Type: application/json" \
  -d '{"text":"Breaking: NASA confirms the Moon landing was staged"}'
```

Max 15,000 characters. 60 requests/hour/IP.

### `POST /api/analyze/url`

```bash
curl -X POST http://localhost:3001/api/analyze/url \
  -H "Content-Type: application/json" \
  -d '{"url":"https://example.com/article"}'
```

- Only `http`/`https` — `file://`, `ftp://`, etc. are rejected.
- Private/loopback/link-local hosts are blocked (SSRF guard, see [Security](#security)).
- Falls back from Jina Reader to a direct fetch with meta-tag extraction.

### `POST /api/detect/text`

Local statistical AI-text detection. **No API key, no network calls, no cost.**

```bash
curl -X POST http://localhost:3001/api/detect/text \
  -H "Content-Type: application/json" \
  -d '{"text":"It is imperative that we leverage innovative solutions to optimize our workflow paradigms."}'
```

Returns 7 metrics (`perplexity`, `burstiness`, `vocabulary`, `repetition`, `sentenceStarts`, `formality`, `punctuation`) plus an overall AI probability. 120 requests/hour/IP.

### `GET /api/face/known`

Returns Face API availability and the list of recognizable identities. Never fails — returns `{"available": false}` when the service is down.

---

## Project structure

```
.
├── server.js                  # Express app: routes, security, upload handling
├── services/
│   ├── groq.js                # Orchestrator: 4-stage image pipeline, text analysis
│   ├── aiDetector.js          # Gemini image forensics + key rotation
│   ├── heuristics.js          # Pure rules: physique, portrait, filename
│   ├── sportsKB.js            # Player↔club knowledge base + jersey matching
│   ├── textDetector.js        # ZeroGPT-style 7-metric statistical detector
│   ├── scraper.js             # Jina/DDG search + URL scrape (SSRF-guarded)
│   ├── retry.js               # Exponential backoff helper
│   └── logger.js              # Structured ISO-timestamped logging
├── python_api.py              # :8000  TF-IDF + Logistic Regression text classifier
├── animal_api.py              # :8001  ONNX AI detection + image classifier
├── face_api.py                # :8002  InsightFace recognition
├── known_faces/               # Reference photos (firstname_lastname.jpg)
├── public/                    # Frontend (no build step)
├── test/                      # node:test suites + Playwright E2E
├── bin/                       # download-face.py, download-models.ps1
├── PROJECT_MAP.md             # Detailed system map + changelog
└── requirements.txt
```

---

## Security

| Concern | Mitigation |
|---|---|
| **SSRF** | `validatePublicUrl()` blocks non-HTTP schemes and any host resolving to private, loopback, link-local, or CGNAT addresses. **Fails closed**, re-validated on every redirect hop (`redirect: 'manual'`), so a hostile server can't bounce the fetcher onto `127.0.0.1` or cloud metadata endpoints. |
| **Malicious uploads** | Multer MIME filter **plus** server-side magic-byte sniffing — the client-declared content type is never trusted. 5 MB cap, temp file deleted in a `finally` block. |
| **XSS** | `sanitize()` strips HTML tags and control characters from all text/URL input. The frontend uses `textContent`/`createElement`, never raw `innerHTML`, for dynamic content. |
| **CSP & headers** | Helmet with a strict policy, `X-Powered-By` hidden, `Permissions-Policy` and `Cross-Origin-Resource-Policy` set. |
| **Rate limiting** | Tiered: 15/hr image (protects API spend), 60/hr analysis, 120/hr local text detector. |
| **API key leakage** | `.env` is gitignored; keys are read only from `process.env` and never returned to clients. Errors are logged server-side, clients get generic messages. |
| **Path traversal / command injection** | No user input reaches `exec()`; the port-in-use handler validates the port as an integer 1–65535. |

---

## Testing

```bash
npm test          # 99 tests (full suite: heuristics, sportsKB, groq, server, scraper, text)
npm run lint      # ESLint
npm run test:unit # core unit suites only (heuristics, sportsKB, groq)
npm run test:e2e  # Playwright (requires a running server)
```

Current suite covers: heuristics (physique/portrait/filename), the sports knowledge base, scraper fallbacks, the SSRF guard, text detection, Groq error paths, and Express integration (upload validation, rate limits, error handling).

CI runs lint + tests on every push and PR to `main` (`.github/workflows/ci.yml`).

---

## Extending

### Add a new face

```bash
npm run download-face -- "Player Name"   # downloads + names the file correctly
# verify the file landed in known_faces/, then restart the Face API
```

Naming convention is `firstname_lastname.jpg`; the API lowercases and converts underscores to spaces (`elon_musk.jpg` → `elon musk`).

For richer analysis, add an entry to `CELEBRITY_CONTEXTS` in `services/sportsKB.js`:

```js
'player name': {
    display: 'Display Name',
    roles: ['role1', 'role2'],
    organizations: ['Org1'],
    typicalSettings: ['setting1'],
    sport: 'football',   // or null
    party: null,         // or 'Democratic' / 'Republican'
    opponents: [],
}
```

### Add a player to the jersey knowledge base

Add an entry to `PLAYER_CLUBS` in `services/sportsKB.js`. A mismatch (photo of a player wearing a team they never played for) then triggers the score-8 hard cap automatically.

### Tune the heuristics

All thresholds live as plain arrays in `services/heuristics.js` — no magic numbers buried in prompts.

---

## Troubleshooting

**Analysis returns 500 / "Failed to analyze image"**
- Confirm `GROQ_API_KEY` is set in `.env` (the server logs a warning at boot if not).
- The image must be JPEG or PNG and under 4 MB. GIF/WebP are rejected by design.
- If the raw log says `invalid image data`, the file is probably corrupt despite its extension.

**Face ID / ML results missing**
- The Python services are optional. Check they're running: `curl http://127.0.0.1:8002/health`.
- The ONNX model must be downloaded: `npm run download-models`.

**Vision models returning 429**
- Rate limited. The system already falls back across 3 models and then Gemini; a 429 across all of them degrades the result rather than failing.

**Port 3001 already in use**
- Change `PORT` in `.env`, or the server will terminate the conflicting process automatically on Windows.

**`npm test` fails to start the server**
- Tests bind to port `0` (a random free port) and skip `.env` loading — they should never conflict with a running instance.

---

## License

No license file has been added yet — treat the code as unlicensed until one is. Add an MIT `LICENSE` file if you intend to keep it open source.

Third-party assets: the reference photos in `known_faces/` are images of public figures, included for local identity-recognition testing only. Replace them with your own licensed images before any commercial deployment.

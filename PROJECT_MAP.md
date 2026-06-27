# PROJECT MAP — Fake Exposer / Fake News Detector

## [TECH_STACK]

| Layer | Technology | Version | Status |
|-------|-----------|---------|--------|
| Frontend | Vanilla HTML/CSS/JS | — | ✅ |
| Backend | Node.js + Express | 4.19.x | ✅ |
| ML Text API | Python + FastAPI + scikit-learn | — | ✅ (port 8000, /health) |
| ML Image API | Python + FastAPI + ONNX Runtime | — | ✅ (port 8001, /health) |
| Face Recognition | Python + FastAPI + InsightFace | buffalo_l | ✅ (port 8002, /health) |
| Text Model | TF-IDF + Logistic Regression | 25k features | ✅ |
| Image Model | MobileNetV2 → ONNX | 4 classes (cat/dog/humans/wild) | ✅ |
| LLM Vision | Groq (LLaMA-4 Scout/Maverick, LLaMA-3.2 Vision) | N/A | ✅ |
| LLM Text | Groq (LLaMA-3.3-70b) | N/A | ✅ |
| AI Forensics | Google Gemini (2.5 Flash, 2.0 Flash) | N/A | ✅ |
| URL Scraping | Jina Reader API | N/A | ✅ |
| Web Search | DuckDuckGo HTML (fallback) | N/A | ⚠️ Fragile (no API key) |

**Service Architecture (refactored):**
```
server.js
├── services/
│   ├── groq.js           (711 lines)  — Core orchestration + Gemini Sports ID
│   ├── heuristics.js     (79 lines)   — Pure rule-based AI/heuristic detection
│   ├── sportsKB.js       (171 lines)  — Player/club knowledge base + jersey matching
│   ├── aiDetector.js     (259 lines)  — Gemini AI image forensics
│   └── scraper.js        (55 lines)   — URL scraping (Jina) + web search (DuckDuckGo)
├── test/
│   ├── heuristics.test.js  — 11 tests (heuristic triggers)
│   └── sportsKB.test.js    — 7 tests (jersey mismatch logic)
└── public/                  — Static frontend (unchanged)
```

**APIs/Keys Required:**
- `GROQ_API_KEY` — text + vision analysis
- `GEMINI_API_KEYS` (CSV array) — AI image detection + sports ID

---

## [SYSTEM_FLOW]

### Text Analysis Flow
```
POST /api/analyze/text { text }
  └─ truncate to 15k chars
  └─ searchWeb() — DuckDuckGo (first 300 chars)
  └─ Groq LLaMA-3.3-70b: fact-check with search context
      └─ POST localhost:8000/predict (TF-IDF + LogReg)
          └─ if fake_prob > 50% → inject ML red flag
          └─ if credibility > 50 + ML says fake → cap to 50
  └─ applyTrustedBoost(): if source verified → +50 pts (cap 100)
  └─ return { credibility_score, verdict, bias, sentiment, red/green flags, summary }
```

### URL Analysis Flow
```
POST /api/analyze/url { url }
  └─ scrapeUrl() — Jina Reader API → markdown text
  └─ Same as Text Analysis (above)
```

### Image Analysis Flow
```
POST /api/analyze/image (multipart: file + optional context)
  └─ Multer: validate MIME type, <5MB, save to uploads/
  └─ FILENAME check (heuristics.js) → immediate 5-score fake if match
  └─ STEP 0: Face API (port 8002 /recognize) → faceIdOverride
  └─ PRE-STEP: Gemini AI Detection (aiDetector.js)
  └─ STEP 1: Groq Vision — describe image
  └─ STEP 2: searchWeb() — DuckDuckGo using description
  └─ STEP 1.5a: detectJerseyMismatch() — local KB (sportsKB.js)
  └─ STEP 1.5b: geminiSportsIdentityCheck() — Gemini (if sports content)
  └─ STEP 2.5: Heuristics (heuristics.js) — physique, portrait, filename
  └─ STEP 3: Groq Vision — full analysis with all context
  └─ POST-STEP-3: Re-check jersey mismatch on LLM output
  └─ MERGE: AI override, jersey mismatch cap, heuristic caps
  └─ FINAL: Image classifier (port 8001 /predict_animal)
  └─ Cleanup: delete temp uploaded file
  └─ applyTrustedBoost() → response
```

### Health Check Flow
```
server.js → every 60s:
  └─ GET localhost:8000/health  → { status, model_loaded }
  └─ GET localhost:8001/health  → { status, model_loaded }
  └─ GET localhost:8002/health  → { status, known_faces_count }
```

---

## [ARCHITECTURE]

### Current (refactored)
```
┌─────────────┐     ┌──────────────────────────────────────────────────┐
│   Browser   │◄───►│           Node.js (port 3001)                    │
│ index.html  │     │  server.js                                       │
│ app.js      │     │  ├── services/groq.js      (orchestrator)        │
└─────────────┘     │  ├── services/heuristics.js (pure rules)         │
                    │  ├── services/sportsKB.js   (KB + matching)      │
                    │  ├── services/aiDetector.js (Gemini forensics)   │
                    │  └── services/scraper.js    (web fetching)       │
                    └────────────────────┬─────────────────────────────┘
                                         │
                  ┌──────────────────────┼──────────────────────┐
                  ▼                      ▼                      ▼
          ┌────────────┐       ┌────────────────┐      ┌──────────────┐
          │ FastAPI    │       │ FastAPI        │      │ FastAPI      │
          │ port 8000  │       │ port 8001      │      │ port 8002    │
          │ TF-IDF+LR  │       │ ONNX MobileNet │      │ InsightFace  │
          │ /predict   │       │ /predict_animal│      │ /recognize   │
          │ /health    │       │ /health        │      │ /health      │
          └────────────┘       └────────────────┘      └──────────────┘
                                         │
                    ┌────────────────────┴────────────────────┐
                    ▼                                         ▼
           ┌───────────────┐                        ┌────────────────┐
           │ Groq API      │                        │ Gemini API     │
           │ LLaMA models  │                        │ Flash models   │
           │ text + vision │                        │ AI detector    │
           └───────────────┘                        │ + sports ID    │
                                                     └────────────────┘
```

---

## [ORPHANS & PENDING]

| Item | Status | Notes |
|------|--------|-------|
| `services/groq.js` → `searchWeb()` | ⚠️ FRAGILE | Scrapes DuckDuckGo HTML. No structured API. HTML changes will break silently. Consider a proper search API key. |
| Rate limiter (15 req/hr) | ⚠️ AGGRESSIVE | Image analysis takes ~30s — 15/hr may be too tight for active users. |
| `start.bat` | ⚠️ FRAGILE | No health check, no restart, multiple cmd windows, no log aggregation. Consider PM2 or a simple Node.js process manager. |
| No CI/CD | ❌ MISSING | No lint, format, or CI pipeline hooks. |
| Duplicate training code | ⚠️ TECH DEBT | `train_model.py`, `train_human_model.py`, `kaggle_script.py` all overlap with `auto_train.py`. Only `auto_train.py` should be kept. |
| `.env` with API keys committed | ⚠️ SECURITY | Keys are in the repo history. Rotate them, add `.env` to `.gitignore`, use env vars. |
| API keys in env as CSV array | ⚠️ FRAGILE | `GEMINI_API_KEYS` comma-separated. Key rotation requires full redeploy. Use a single key + fallback. |

### Resolved Items ✅
| Item | Resolution |
|------|-----------|
| `services/ocr.js` broken | DELETED — unused, broken env var reference |
| `imageCache` + `saveCache()` dead code | REMOVED — undefined `CACHE_FILE` variable |
| `cheerio` unused dep | REMOVED from package.json |
| `tesseract.js` unused dep | REMOVED from package.json |
| No CORS restriction | FIXED — `cors({ origin: false })` same-origin only |
| Face API temp files not cleaned | FIXED — now writes to `uploads/` dir, cleaned by server cleanup job |
| No `/health` endpoints | ADDED — all 3 Python APIs expose `/health` |
| No health check polling | ADDED — server polls every 60s |
| No test suite | ADDED — 18 unit tests, `npm test` / `npm run test:unit` |
| `groq.js` monolith (1166 lines) | REFACTORED → 711 lines + extracted heuristics (79), sportsKB (171), web search into scraper (55) |
| No requirements.txt | ADDED — `requirements.txt` with pinned Python deps |
| Startup exits on missing GROQ_API_KEY | FIXED — warning+degradation instead of `process.exit(1)`. Groq endpoints return clear error. 2 tests added. |

---

## Milestones (Verifiable Goals)

| # | Milestone | Status | Verification |
|---|-----------|--------|-------------|
| M1 | Split `services/groq.js` into domain modules | ✅ DONE | 1166→711 lines. Modules: heuristics, sportsKB, web search extracted. |
| M2 | Fix/remove dead code | ✅ DONE | ocr.js deleted, imageCache removed, cheerio+tesseract.js removed from pkg. |
| M3 | Pin Python dependencies | ✅ DONE | `requirements.txt` generated from installed packages. |
| M4 | Move API keys out of repo | ⚠️ PENDING | Alert: `.env` still in repo. User must rotate keys and add to `.gitignore`. |
| M5 | Add process health checks | ✅ DONE | `/health` on all APIs + 60s polling in server.js. |
| M6 | Add test suite | ✅ DONE | 20 unit tests + 4 integration (skip offline). `npm test` / `npm run test:unit`. |
| M7 | Generate requirements.txt | ✅ DONE | `requirements.txt` covers all Python deps. |

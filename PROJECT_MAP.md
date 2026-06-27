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
│   ├── groq.js           (664 lines)  — Core orchestration
│   ├── heuristics.js     (79 lines)   — Pure rule-based AI/heuristic detection
│   ├── sportsKB.js       (171 lines)  — Player/club knowledge base + jersey matching
│   ├── aiDetector.js     (259 lines)  — Gemini AI image forensics
│   └── scraper.js        (120 lines)  — URL scraping (Jina) + web search (DDG + Jina fallback)
├── test/
│   ├── heuristics.test.js  — 11 tests
│   ├── sportsKB.test.js    — 12 tests
│   ├── scraper.test.js     — 7 tests
│   ├── groq.test.js        — 2 tests
│   └── server.test.js      — 7 integration tests (auto-starts Express)
└── public/                  — Static frontend (updated: progress bar)
```

**APIs/Keys Required:**
- `GROQ_API_KEY` — text + vision analysis
- `GEMINI_API_KEY` — AI image forensics

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
  └─ STEP 1.5: detectJerseyMismatch() — local KB (sportsKB.js)
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
+-------------+     +----------------------------------------------------+
|   Browser   |<-->|           Node.js (port 3001)                      |
| index.html  |    |  server.js                                         |
| app.js      |    |  +-- services/groq.js      (orchestrator)           |
+-------------+    |  +-- services/heuristics.js (pure rules)            |
                   |  +-- services/sportsKB.js   (KB + matching)         |
                   |  +-- services/aiDetector.js (Gemini forensics)      |
                   |  +-- services/scraper.js    (web fetching)          |
                   +---------------------------+------------------------+
                                               |
            +----------------------------------+--------------------+
            |                                  |                    |
            v                                  v                    v
    +---------------+                +------------------+    +--------------+
    | FastAPI       |                | FastAPI          |    | FastAPI      |
    | port 8000     |                | port 8001        |    | port 8002    |
    | TF-IDF+LR     |                | ONNX MobileNet   |    | InsightFace  |
    | /predict      |                | /predict_animal  |    | /recognize   |
    | /health       |                | /health          |    | /health      |
    +---------------+                +------------------+    +--------------+
                                               |
                    +--------------------------+---------------------------+
                    |                                                      |
                    v                                                      v
           +-------------------+                                 +------------------+
           | Groq API          |                                 | Gemini API       |
           | LLaMA models      |                                 | Flash models     |
           | text + vision     |                                 | AI forensics     |
           +-------------------+                                 +------------------+
```

---

## [ORPHANS & PENDING]

*None — all identified issues have been resolved.*

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
| No test suite | ADDED — 38 tests (31 unit + 7 integration). `npm test` runs all. Integration tests auto-start the Express server. |
| `groq.js` monolith (1166 lines) | REFACTORED → 711 lines + extracted heuristics (79), sportsKB (171), web search into scraper (55) |
| No requirements.txt | ADDED — `requirements.txt` with pinned Python deps |
| Startup exits on missing GROQ_API_KEY | FIXED — warning+degradation instead of `process.exit(1)`. Groq endpoints return clear error. 2 tests added. |
| API keys leaked in repo | FIXED — user rotated keys. No keys were in git history (.env was already gitignored). Old keys removed from `.env`. |
| `GEMINI_API_KEYS` as CSV array | FIXED — reduced to single key. |
| Rate limiter (15->60 req/hr) | BUMPED — `windowMs: 60min, max: 60`. Adequate for interactive image analysis (~30s/req). |
| Duplicate training scripts | DELETED — `train_model.py`, `train_human_model.py`, `kaggle_script.py` all removed; `auto_train.py` covers both text + image domains. |
| Face API not spawned | FIXED — `face_api.py` now auto-launched by server.js with health checks and graceful shutdown. |
| `start.bat` fragile | REWRITTEN — single-window launcher with prereq checks, auto `npm install`, auto `pip install`, error handling. Server orchestrates all Python APIs internally. |
| `searchWeb()` DuckDuckGo scraper fragile | HARDENED — 4 falling regex patterns + lite endpoint fallback + better text cleaning. |
| No CI/CD | ADDED — `.github/workflows/ci.yml` runs `npm test` on push/PR. |
| Frontend | AUDITED — no bugs found. Clean HTML/CSS/JS. |
| No README | ADDED — `README.md` with setup, architecture, scripts reference. |
| No scraper tests | ADDED — 7 tests for `searchWeb` + `scrapeUrl` with mocked fetch. |
| Gemini Sports identity check unreliable | REMOVED — `geminiSportsIdentityCheck()` deleted from groq.js. Sports identity now relies solely on visible jersey text (name/number) + local KB mismatch detection. Player facial recognition from AI was frequently wrong. |
| SportsKB coverage gap | FIXED — added Frenkie de Jong and Michael Olise aliases to player database. 12 sportsKB tests total. |
| Face API multipart upload broken | FIXED — replaced built-in `FormData()` with `form-data` npm package for reliable multipart uploads to Python face microservice. |
| Image analysis serial (Face → Gemini → Step 1) | FIXED — now runs Face API + Gemini + Step 1 in parallel via `Promise.all()`, saving ~10s. |
| Web search in image path (slow + unreliable) | REMOVED — `searchWeb()` call removed from image analysis. It's still used in text/URL paths. Added Jina search fallback (`s.jina.ai`) for text path. |
| No gzip compression | FIXED — added `compression()` middleware. ~70% smaller HTML/CSS/JS payloads. |
| No static asset caching | FIXED — added `maxAge: '7d'` to `express.static`. Browser caches frontend files. |
| No integration tests | FIXED — `test/server.test.js` now auto-starts Express on a random port and tests static serving, validation (400s), CORS, 404s, compression, and cache headers. 7 integration tests. |
| Frontend has no loading feedback | FIXED — added step-based progress bar with animated indicators per analysis type (text/URL/image). |

---

## Milestones (Verifiable Goals)

| # | Milestone | Status | Verification |
|---|-----------|--------|-------------|
| M1 | Split `services/groq.js` into domain modules | ✅ DONE | 1166->711 lines. Modules: heuristics, sportsKB, web search extracted. |
| M2 | Fix/remove dead code | ✅ DONE | ocr.js deleted, imageCache removed, cheerio+tesseract.js removed from pkg. |
| M3 | Pin Python dependencies | ✅ DONE | `requirements.txt` generated from installed packages. |
| M4 | Rotate API keys and secure repo | ✅ DONE | Keys rotated. `.env` was already gitignored — no keys in history. Single key per service. |
| M5 | Add process health checks | ✅ DONE | `/health` on all APIs + 60s polling in server.js. |
| M6 | Add test suite | ✅ DONE | 38 tests (31 unit + 7 integration). `npm test` runs all. Integration tests auto-start Express. |
| M7 | Generate requirements.txt | ✅ DONE | `requirements.txt` covers all Python deps. |
| M8 | Consolidate duplicate training scripts | ✅ DONE | Deleted `train_model.py`, `train_human_model.py`, `kaggle_script.py`. All covered by `auto_train.py`. |
| M9 | Single-window process manager | ✅ DONE | Face API auto-spawned. `start.bat` rewritten with prereq checks. Single `npm start` runs everything. |
| M10 | CI/CD pipeline | ✅ DONE | `.github/workflows/ci.yml` — runs tests on push/PR. |
| M11 | README + scraper tests | ✅ DONE | `README.md` with full docs. 7 new scraper tests. |
| M12 | Remove unreliable Gemini sports facial recognition | ✅ DONE | `geminiSportsIdentityCheck()` removed. Step 1 prompt changed to only ID player from visible jersey text. Step 3 fact-check simplified. Face API + local KB + visible text are the only identity sources. |
| M13 | Performance optimization (zero-cost) | ✅ DONE | Parallelized Face API + Gemini + Groq Step 1 (~10s saved). Removed `searchWeb()` from image path (~5-10s saved). Groq Step 1 tokens 500→250. gzip compression + 7d static cache. |

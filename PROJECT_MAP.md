# PROJECT MAP — Fake Exposer / Fake News Detector

## [TECH_STACK]

| Layer | Technology | Version | Status |
|-------|-----------|---------|--------|
| Frontend | Vanilla HTML/CSS/JS | — | ✅ |
| Backend | Node.js + Express | 4.22.x | ✅ |
| ML Text API | Python + FastAPI + scikit-learn | — | ✅ (port 8000) |
| ML Image API | Python + FastAPI + ONNX Runtime | — | ✅ (port 8001) |
| Face Recognition | Python + FastAPI + InsightFace | buffalo_l | ✅ (port 8002) |
| Text Model | TF-IDF + Logistic Regression | 25k features | ❌ Untrained (no data) |
| Image Model | MobileNetV2 → ONNX | 4 classes (cat/dog/humans/wild) | ❌ Untrained (no data) |
| LLM Vision | Groq (LLaMA-4 Scout/Maverick, LLaMA-3.2 Vision) | — | ✅ |
| LLM Text | Groq (LLaMA-3.3-70b) | — | ✅ |
| AI Forensics | Google Gemini (2.5 Flash, 2.0 Flash) | — | ✅ |
| URL Scraping | Jina Reader API | — | ✅ |
| Web Search | DuckDuckGo HTML (fallback) | — | ⚠️ Fragile |

**Service Architecture (refactored):**
```
server.js
├── services/
│   ├── retry.js          (27 lines)   — `withRetry()` exponential-backoff utility
│   ├── groq.js           (760 lines)  — Core orchestration
│   ├── heuristics.js     (79 lines)   — Pure rule-based AI/heuristic detection
│   ├── sportsKB.js       (195 lines)  — Player/club knowledge base + jersey matching + findTeamInText()
│   ├── aiDetector.js     (303 lines)  — Gemini AI image forensics (retry added)
│   ├── textDetector.js   (400+ lines) — ZeroGPT-style AI text detector (7-metric ensemble)
│   └── scraper.js        (135 lines)  — URL scraping (Jina) + web search (DDG + Jina fallback)
├── nonescape-mini-v0.onnx  — (83 MB)  Nonescape AI image detection ONNX model (Apache 2.0)
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
  └─ PARALLEL (Promise.all):
  │   ├─ Face API (port 8002 /recognize) → faceIdOverride
  │   ├─ Gemini AI Detection (aiDetector.js)
  │   ├─ Nonescape ONNX (port 8001 /detect_ai) → ai probability
  │   └─ STEP 1: Groq Vision — describe image (250 tokens)
  ├─ searchWeb() removed from image path
  ├─ STEP 1.5: detectJerseyMismatch() on description + userContext + faceId
  ├─ STEP 1.5b: findTeamInText() fallback — unverifiable-identity cap
  ├─ STEP 2.5: Heuristics (heuristics.js) — physique, portrait, filename
  └─ STEP 3: Groq Vision — full analysis with all context + nonescapeNote
  └─ POST-STEP-3: Re-check jersey mismatch on LLM output + userContext
  └─ MERGE: AI override, Nonescape cap, jersey mismatch cap, heuristic caps, unverifiable cap
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

### Current Issues
| Item | Location | Impact | Action Needed |
|------|----------|--------|---------------|
| `nonescape-mini-v0.onnx` missing | `animal_api.py:22` | `/detect_ai` degraded | Run `npm run download-models` or download manually from GitHub |
| `WELFake_Dataset.csv` missing | Root | Text model not trained | Download from Kaggle and place in root |
| `archive.zip` / `humans.zip` missing | Root | Image model not trained | Download AFHQ from Kaggle + human faces zip |

### Resolved Items ✅
| Item | Resolution |
|------|-----------|
| `auto_train.py` UnicodeEncodeError on Windows | FIXED — replaced all non-ASCII chars (`─`, `→`, `…`, `—`, emoji) in `print()` calls with ASCII. No more cp1252 crash. |
| `auto_train.py` NUM_EPOCHS=1 (garbage model) | FIXED — bumped to 10 epochs |
| `eslint.config.js` invalid rule `preserve-caught-error` | REMOVED — not a real ESLint rule, silently ignored |
| `start.bat` only checks `fastapi` pip package | FIXED — now runs `pip install -r requirements.txt` unconditionally |
| `services/ocr.js` broken | DELETED — unused, broken env var reference |
| `imageCache` + `saveCache()` dead code | REMOVED — undefined `CACHE_FILE` variable |
| `cheerio` unused dep | REMOVED from package.json |
| `tesseract.js` unused dep | REMOVED from package.json |
| No CORS restriction | FIXED — `cors({ origin: false })` same-origin only |
| Face API temp files not cleaned | FIXED — now writes to `uploads/` dir, cleaned by server cleanup job |
| No `/health` endpoints | ADDED — all 3 Python APIs expose `/health` |
| No health check polling | ADDED — server polls every 60s |
| No test suite | ADDED — 39 tests. `npm test` runs all. Integration tests auto-start Express. |
| `groq.js` monolith (1166 lines) | REFACTORED → 711 lines + extracted heuristics, sportsKB, scraper |
| No requirements.txt | ADDED — pinned Python deps |
| Startup exits on missing GROQ_API_KEY | FIXED — warning+degradation, no `process.exit(1)` |
| API keys leaked in repo | FIXED — user rotated keys. `.env` was already gitignored. |
| Rate limiter (15->60 req/hr) | BUMPED — adequate for interactive use |
| Duplicate training scripts | DELETED — `train_model.py`, `train_human_model.py`, `kaggle_script.py` removed |
| Face API not spawned | FIXED — auto-launched by server.js |
| `start.bat` fragile | REWRITTEN — single-window launcher |
| `searchWeb()` DuckDuckGo scraper fragile | HARDENED — 4 regex fallbacks + lite endpoint |
| No CI/CD | ADDED — `.github/workflows/ci.yml` |
| No README | ADDED |
| No scraper tests | ADDED — 7 tests |
| Gemini sports facial recognition unreliable | REMOVED — relies on visible jersey text + local KB |
| Face API multipart upload broken | FIXED — uses `form-data` npm package |
| Image analysis serial (slow) | FIXED — parallel `Promise.all()` saves ~10s |
| Web search in image path (slow) | REMOVED — text/URL only |
| No gzip compression | FIXED — `compression()` middleware |
| No static cache | FIXED — `maxAge: '7d'` |
| No integration tests | FIXED — 7 tests |
| No loading feedback | FIXED — progress bar |
| No retry for transient failures | FIXED — `services/retry.js` + applied to Groq, Gemini, Jina |
| TDZ ReferenceError in auto-trainer | FIXED |
| isMainModule broken on Windows spaces | FIXED |
| Duplicate dynamic import('form-data') | FIXED — hoisted |
| searchJina silent errors | FIXED — added `console.warn` |
| server.test.js close not awaited | FIXED |
| No Nonescape AI classifier | ADDED — /detect_ai endpoint |
| Jersey mismatch misses face-swapped fakes | FIXED — Step 1 prompt, userContext scan, unverifiable fallback |
| No ZeroGPT-style AI text detector | ADDED — standalone /api/detect/text with 7 metrics |

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
| M14 | Retry logic + reliability polish | ✅ DONE | `withRetry()` utility with exponential backoff. Applied to: Groq text analysis, Groq vision model timeouts, Gemini model fetch, Jina URL scraping. Also: hoisted `form-data` to top-level import, fixed `isMainModule` URL-encoding on Windows, fixed TDZ `ReferenceError` in auto-trainer, made `server.close()` async in tests, `.unref()` on all timers. |
| M15 | Nonescape local AI detection model | ✅ DONE | Downloaded NonescapeClassifierMini (EfficientNet, 82.7MB, Apache 2.0), exported to ONNX. Added `/detect_ai` endpoint to `animal_api.py`. Runs in parallel with Face API + Gemini + Groq Step 1. Score caps at 30 if >65% AI probability. |
| M16 | Jersey mismatch + identity fallback | ✅ DONE | Relaxed Step 1 prompt: LLM MUST name ultra-famous players from face (not just jersey text). Jersey mismatch check now scans `userContext` + `faceIdOverride` too. Added `findTeamInText()` and unverifiable-identity fallback (cap at 35 when team found but no player name in any source). |
| M17 | ZeroGPT-style AI text detector | ✅ DONE | \services/textDetector.js\ — 7-metric multi-stage ensemble: formality (colloquial/AI-vocab ratio), perplexity (word frequency rarity), burstiness (sentence length CV), vocabulary diversity, repetition (bigram + transition density), sentence start diversity, punctuation diversity. No API key needed. Exposed at \POST /api/detect/text\ with sentence-level highlighting UI. Integrated into \analyzeContent()\ pipeline as pre-analysis context and post-hoc score cap. |
| M18 | Fix Windows cp1252 Unicode crash in auto_train.py | ✅ DONE | Replaced all non-ASCII chars in `print()` calls with ASCII. Auto-trainer no longer crashes on Windows. |
| M19 | Bump image training epochs | ✅ DONE | NUM_EPOCHS 1 → 10 for meaningful MobileNetV2 fine-tuning |
| M20 | Fix start.bat pip install | ✅ DONE | Now runs `pip install -r requirements.txt` unconditionally instead of only checking for `fastapi` |
| M21 | Remove invalid ESLint rule | ✅ DONE | Removed non-existent `preserve-caught-error` from eslint config |
| M22 | Add nonescape model download hint | ✅ DONE | Clear HuggingFace URL printed when model fails to load |
| M23 | Add JSON 404 handler for API routes | ✅ DONE | `app.use('/api/*')` returns `{error: "Not found: ..."}` instead of HTML |
| M24 | Add Python API process watchdog | ✅ DONE | Auto-restarts Python APIs on crash (3s delay). Uses `spawnPythonApi()` helper. |
| M25 | Create download script for ML models | ✅ DONE | `bin/download-models.ps1` + `npm run download-models` |
| M26 | Clean auto_train.py docstring and comments | ✅ DONE | Replaced all box-drawing chars with ASCII |
| M27 | Delete orphan bin/download-face.js | ✅ DONE | Package.json points to Python version only |
| M28 | Fix download_faces.ps1 path | ✅ DONE | Uses `$PSScriptRoot` instead of hardcoded path |
| M29 | Fix start.bat pip check | ✅ DONE | Runs `pip install -r requirements.txt` unconditionally |
| M30 | Remove invalid ESLint rule | ✅ DONE | Removed `preserve-caught-error` (not a real rule) |
| M31 | Add 404 JSON tests | ✅ DONE | 2 new tests verify JSON error format for unknown API routes |
| M32 | Fix misleading "All models trained" message | ✅ DONE | `auto_train.py` exits 1 when no data found. `server.js` prints honest message and skips API restart. No more crash loop. |
| M33 | Increase Face API health check retries | ✅ DONE | Changed from 3x5s to 6x7s giving Face API up to ~65s to initialize (was ~35s). Reduces false \"unreachable\" reports. |

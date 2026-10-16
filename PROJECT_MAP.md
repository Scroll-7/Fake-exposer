# PROJECT MAP — Fake Exposer / Fake News Detector

## [TECH_STACK]

| Layer | Technology | Version | Status |
|-------|-----------|---------|--------|
| Frontend | Vanilla HTML/CSS/JS | — | ✅ |
| Backend | Node.js + Express | 4.x | ✅ |
| ML Text API | Python + FastAPI + scikit-learn | — | ✅ (port 8000) |
| ML Image API | Python + FastAPI + ONNX Runtime | — | ✅ (port 8001) |
| Face Recognition | Python + FastAPI + InsightFace | buffalo_l | ✅ (port 8002) |
| Text Model | TF-IDF + Logistic Regression | 25k features | ✅ Trained — leak-free, calibrated (in-domain ROC-AUC 0.991; does not transfer to modern text) |
| Image Model (animal) | MobileNetV2 → ONNX | — | ❌ REMOVED — no fake-news signal |
| LLM Vision | Groq (LLaMA-4 Scout/Maverick via direct `fetch()`) | — | ✅ (SDK bypassed) |
| LLM Text | Groq (LLaMA-3.3-70b via SDK) | — | ✅ |
| AI Forensics | Google Gemini (2.5 Flash, 2.0 Flash) | — | ✅ |
| AI Image Detection | Nonescape mini v0 ONNX | — | ✅ (valid local model; v1 re-trained head is opt-in via `NONESCAPE_MODEL`) |
| URL Scraping | Jina Reader API | — | ✅ |
| Web Search | DuckDuckGo HTML (fallback) | — | ⚠️ Fragile |

**Service Architecture:**
```
server.js (399 lines)
├── services/
│   ├── groq.js          (1060 lines) — Core orchestrator + LLM calls + fallback
│   ├── textDetector.js  (470 lines)  — ZeroGPT-style 7-metric ensemble
│   ├── aiDetector.js    (274 lines)  — Gemini AI image forensics
│   ├── sportsKB.js      (384 lines)  — Player/club KB + jersey matching
│   ├── scraper.js       (262 lines)  — URL scraping (Jina) + DDG search
│   ├── heuristics.js    (80 lines)   — Pure rules: physique, portrait, filename
│   ├── retry.js         (28 lines)   — Exponential backoff utility
│   └── logger.js        (21 lines)   — Structured logging (ISO ts + 4 levels)
├── python APIs (optional — run manually, not auto-spawned):
│   ├── python_api.py    (55 lines)   — Text classification (calibrated TF-IDF + LR)
│   ├── image_api.py     (65 lines)   — Nonescape ONNX AI-image detection
│   └── face_api.py      (102 lines)  — InsightFace recognition
├── test/  (172 tests total)
│   ├── sportsKB.test.js    — 49 tests
│   ├── scraper.test.js     — 30 tests
│   ├── groq.test.js        — 22 tests
│   ├── aiDetector.test.js  — 18 tests
│   ├── server.test.js      — 17 integration tests
│   ├── heuristics.test.js  — 11 tests
│   ├── logger.test.js      — 10 tests
│   ├── retry.test.js       — 8 tests
│   └── textDetector.test.js — 7 tests
├── eval/                    — Harness: dedupe, split, metrics, report, image_eval,
│                              fetch_cifake, make_split (32 self-tests, no data)
└── public/                  — Static frontend (Text, URL, Image tabs)
```

**APIs/Keys Required:**
- `GROQ_API_KEY` — text + vision analysis
- `GEMINI_API_KEYS` — AI image forensics (comma-separated for rotation)

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
  └─ FILENAME check (checkAiFilename) → immediate fake verdict (score=5)
  └─ PARALLEL (Promise.all):
  │   ├─ Face API (port 8002 /recognize) → faceIdOverride
  │   ├─ Gemini AI Detection (aiDetector.js) → aiProbability
  │   ├─ Nonescape ONNX (port 8001 /detect_ai) → aiScore (degraded if model missing)
  │   └─ Groq Vision Step 1 — describe image (250t, direct fetch)
  ├─ detectJerseyMismatch() on description + userContext + faceId
  ├─ findTeamInText() fallback → unverifiable-identity cap
  ├─ Heuristics (heuristics.js) — physique, portrait, filename
  ├─ Groq Vision Step 3 — full analysis with all context + nonescapeNote
  ├─ POST-STEP-3: Re-check jersey mismatch on LLM output + userContext
  ├─ MERGE: AI override, Nonescape high-confidence cap (≥0.9), jersey mismatch cap,
  │         heuristic caps, unverifiable cap, portrait penalty
  ├─ ML confidence gate: local text model caps at 50 only when calibrated p(fake) ≥ 0.8
  ├─ Cleanup: delete temp file
  └─ applyTrustedBoost() → response
```

### AI Text Detection Flow (no API key needed)
```
POST /api/detect/text { text }
  └─ tokenize → words, splitSentences → sentences
  └─ 7 metrics (all 0-100, higher = more AI-like):
  │   ├─ perplexity       — word frequency rarity (low = common words = AI)
  │   ├─ burstiness       — sentence length CV (low = uniform = AI)
  │   ├─ vocabulary       — lexical diversity + TTR
  │   ├─ repetition       — bigram repetition + transition density + stopword density
  │   ├─ sentenceStarts   — diversity of sentence beginnings + AI starter ratio
  │   ├─ formality        — colloquial density (low = formal = AI) + transition density
  │   └─ punctuation      — varied punctuation density (low = standard only = AI)
  └─ weighted ensemble (perplexity=0.15, burstiness=0.10/0.20, vocab=0.15,
  │                      repetition=0.15, sentenceStarts=0.15, formality=0.25,
  │                      punctuation=0.10) ÷ totalWeight
  └─ shortTextPenalty: <10 words=0.3, <20=0.5, <40=0.75, else=1.0
  └─ sentence-level highlighting (per-sentence AI score ≥55 = flagged)
  └─ return { overallScore, 7 metrics, sentences, aiSentencePercentage }
```

---

## [ARCHITECTURE]

```
+-------------+     +----------------------------------------------------+
|   Browser   |<-->|           Node.js (port 3001)                      |
| index.html  |    |  server.js                                         |
| app.js      |    |  +-- services/groq.js      (orchestrator, 1060L)    |
+-------------+    |  +-- services/textDetector.js (AI text, 470L)      |
                   |  +-- services/aiDetector.js (Gemini, 274L)         |
                   |  +-- services/sportsKB.js   (KB + matching, 384L)  |
                   |  +-- services/scraper.js    (web fetching, 262L)   |
                   |  +-- services/heuristics.js (pure rules, 80L)      |
                   |  +-- services/retry.js      (utility, 28L)         |
                   |  +-- services/logger.js     (logging, 21L)         |
                   +---------------------------+------------------------+
                                               |
             +---------------------------------+-------------------+
             |                                 |                   |
             v                                 v                   v
     +----------------+              +------------------+  +--------------+
     | FastAPI        |              | FastAPI          |  | FastAPI      |
     | port 8000      |              | port 8001        |  | port 8002    |
     | TF-IDF+LR      |              | ONNX Nonescape   |  | InsightFace  |
     | /predict       |              | /detect_ai       |  | /recognize   |
     | /health        |              | /detect_ai       |  | /health      |
     +----------------+              +------------------+  +--------------+
                                               |
                    +--------------------------+--------------------------+
                    |                                                     |
                    v                                                     v
           +-------------------+                                +------------------+
           | Groq API          |                                | Gemini API       |
           | LLaMA models      |                                | Flash models     |
           | text + vision     |                                | AI forensics     |
           +-------------------+                                +------------------+
```

---

## [ORPHANS & PENDING]

### Current Issues (External Blockers)
| Item | Location | Impact | Action Needed |
|------|----------|--------|---------------|
| `WELFake_Dataset.csv` not in root | Root | Cannot re-run `eval.report` or retrain the text model (the shipped model still works) | Download from Kaggle — see `eval/README.md` |
| Sports-realistic labelled set | — | The only distribution that predicts production behaviour | Build ~50 images/class with provenance labels (see `eval/README.md`, Phase 3) |

### Resolved Items ✅
| Item | Resolution |
|------|-----------|
| Groq vision SDK serialization bug (file paths instead of base64) | FIXED — replaced `groq-sdk` vision call with direct `fetch()` to `https://api.groq.com/openai/v1/chat/completions` |
| Groq vision crash when API is down | FIXED — `analyzeImage` catch block returns degraded result from Gemini + heuristics + Face API + Nonescape instead of throwing |
| `meta-llama/llama-4-maverick-17b-128e-instruct` not in model list | ADDED — second vision fallback model |
| `qwen/qwen3.6-27b` in VISION_MODELS (no vision support) | REMOVED |
| Portrait heuristic false positives on real selfies | FIXED — merged to require BOTH portrait framing AND perfect-quality signals (smooth skin, studio lighting), not just portrait alone |
| Portrait heuristic hard cap (score=15 override) | SOFTENED — changed to ~55% penalty reduction, preserves Groq verdict when adjusted score >30 |
| AI Detector tab in frontend | REMOVED — HTML tab button, tab-pane, JS logic, CSS styles all deleted |
| Formality weight mismatch (0.25 numerator vs 0.20 denominator) | FIXED — denominator changed from 0.20 to 0.25 |
| `groq-sdk` dependency | REMOVED — text and vision both use raw REST `fetch()` calls with key failover |
| Morgan logging too verbose (`'combined'`) | CHANGED to `'short'` |
| Python API stdout noise | SUPPRESSED — stdout forwarded nowhere, stderr only logs errors |
| Auto-trainer startup logs noisy | QUIETED — `stdio: 'ignore'`, only logs on completion |
| Health check polling success log | REMOVED — only logs on failure |
| `auto_train.py` UnicodeEncodeError on Windows | FIXED — replaced non-ASCII chars in `print()` calls with ASCII |
| `auto_train.py` NUM_EPOCHS=1 (garbage model) | FIXED — bumped to 10 epochs |
| `eslint.config.js` invalid rule `preserve-caught-error` | REMOVED |
| `start.bat` only checks `fastapi` pip package | FIXED — runs `pip install -r requirements.txt` unconditionally |
| `services/ocr.js` broken | DELETED |
| `imageCache` + `saveCache()` dead code | REMOVED |
| `cheerio` unused dep | REMOVED from package.json |
| `tesseract.js` unused dep | REMOVED from package.json |
| No CORS restriction | FIXED — `cors({ origin: false })` same-origin only |
| Face API temp files not cleaned | FIXED — writes to `uploads/`, cleaned by server cleanup job |
| No `/health` endpoints | ADDED — all 3 Python APIs |
| No health check polling | ADDED — server polls every 60s |
| No test suite | ADDED — 39 tests. `npm test` runs all. |
| `groq.js` monolith (1166 lines) | REFACTORED → 711 lines + extracted modules; now 975 lines after adding features |
| No requirements.txt | ADDED — pinned Python deps |
| Startup exits on missing GROQ_API_KEY | FIXED — warning+degradation, no `process.exit(1)` |
| API keys leaked in repo | FIXED — rotated. `.env` gitignored. |
| Rate limiter (15→60 req/hr) | BUMPED |
| Duplicate training scripts | DELETED — `train_model.py`, `train_human_model.py`, `kaggle_script.py` |
| Face API not spawned | FIXED — auto-launched by server.js |
| `start.bat` fragile | REWRITTEN — single-window launcher |
| `searchWeb()` DuckDuckGo scraper fragile | HARDENED — 4 regex fallbacks + lite endpoint |
| No CI/CD | ADDED — `.github/workflows/ci.yml` + `test.yml` |
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
| No Nonescape AI classifier | ADDED — `/detect_ai` endpoint |
| Jersey mismatch misses face-swapped fakes | FIXED — Step 1 prompt, userContext scan, unverifiable fallback |
| No ZeroGPT-style AI text detector | ADDED — standalone `/api/detect/text` with 7 metrics |
| `nonescape-mini-v0.onnx` missing | DOWNLOADED — 82.7 MB via `npm run download-models` |
| All `console.*` calls unstructured | FIXED — `services/logger.js` with ISO timestamps + 4 levels (ERROR/WARN/INFO/DEBUG). All 63 console calls replaced across 4 files. |
| Auto-trainer spawn hangs tests | FIXED — guarded by `NODE_ENV !== 'test'` |
| `.env.example` missing | CREATED |
| `groq.js` size (975 lines) | ASSESSED — coherent monolith (all Groq LLM logic). Split deferred (low ROI, high regression risk). |
| `WORD_FREQ` 5 duplicate keys in `textDetector.js` | REMOVED — `important(2.07)`, `environment(1.65)`, `significant(1.15)`, `practice(1.37)`, `various(1.01)` — first occurrence deleted, only the later value remains (no behavior change). |
| No explicit Content-Security-Policy | FIXED — Helmet CSP configured: `default-src 'self'`, Google Fonts+Fonts allowed, `script-src 'self'`, `img-src 'self' data:`, `frame-ancestors 'none'`, `object-src 'none'`. |
| Image 500 error leaks `error.message` to client | FIXED — now returns generic `'Failed to analyze image'`. |
| No server-side text/URL input length cap | FIXED — `/api/analyze/text` rejects >15k chars, `/api/analyze/url` rejects >5k chars. |
| `X-Powered-By: Express` header leaks server identity | FIXED — `hidePoweredBy: true` in helmet config. |
| No `Permissions-Policy` header | FIXED — disabled camera, mic, geolocation, usb, bluetooth, midi, payment, sensors. |
| No `Cross-Origin-Resource-Policy` header | FIXED — set to `same-origin`. |
| `PORT` shell injection in EADDRINUSE handler | FIXED — port validated as integer 1–65535 before use in `exec()`. |
| XSS vector via `innerHTML` in face badge | FIXED — uses `textContent` + `createElement` instead. |
| HTML tags accepted in text inputs | FIXED — `sanitize()` strips `<...>` tags from all text/URL inputs. |
| No HTTPS redirect | FIXED — middleware redirects HTTP→HTTPS (301), skips localhost and requests behind proxy via `X-Forwarded-Proto`. |
| No magic byte validation on uploaded images | FIXED — `validateImageMagicBytes()` checks JPEG/PNG/GIF/WEBP file signatures server-side; client MIME type alone no longer trusted. |
| Image endpoint shares same generous rate limit (60/hr) | FIXED — image endpoint has its own 15/hr limit to protect API costs. |
| No rate limit on `/api/detect/text` | FIXED — 120/hr limit applied. |
| Control characters / null bytes in text inputs | FIXED — `sanitize()` strips `\x00-\x08\x0B\x0C\x0E-\x1F\x7F` from all text/URL inputs. |
| Vision model retried on ANY error (incl. 400) | FIXED — `groqVisionRequest` now treats 429/503/404/no-vision as transient (try next model) and aborts immediately on non-transient errors (e.g. 400 invalid image) instead of burning all model calls. |
| GIF/WebP uploads silently degraded to "unable to analyze" | FIXED — uploads restricted to JPEG/PNG (multer + magic bytes + MIME map). Clear 400 message returned for GIF/WebP/unknown types. |
| SSRF via `/api/analyze/url` (scrapeDirect) | FIXED — `validatePublicUrl()` blocks non-http(s) schemes and any host resolving to private/loopback/link-local IPs (fail-closed, per redirect hop, `redirect: 'manual'`). Plus scheme guard in the URL route. |
| Filename instant-verdict false positives (generic words) | FIXED — server's wide `AI_FILENAME_KEYWORDS` (`fake`, `edited`, `enhanced`, `photoshop`, `gpt`, `openai`, `generated`, …) removed. Both hard-verdict paths now share the strict list exported from `heuristics.js` (only unambiguous AI tool names). |
| Docs claim Python APIs auto-spawn (server no longer does) | FIXED — README + start.bat + PROJECT_MAP updated to reflect manual run. |

---

## Milestones (Verifiable Goals)

| # | Milestone | Status | Verification |
|---|-----------|--------|-------------|
| M1 | Split `services/groq.js` into domain modules | ✅ DONE | 1166→711 lines. Modules: heuristics, sportsKB, web search extracted. |
| M2 | Fix/remove dead code | ✅ DONE | ocr.js deleted, imageCache removed, cheerio+tesseract.js removed |
| M3 | Pin Python dependencies | ✅ DONE | `requirements.txt` generated |
| M4 | Rotate API keys and secure repo | ✅ DONE | Keys rotated. `.env` gitignored |
| M5 | Add process health checks | ✅ DONE | `/health` on all APIs + 60s polling |
| M6 | Add test suite | ✅ DONE | 39 tests. `npm test` runs all |
| M7 | Generate requirements.txt | ✅ DONE | Covers all Python deps |
| M8 | Consolidate duplicate training scripts | ✅ DONE | Only `auto_train.py` remains |
| M9 | Single-window process manager | ✅ DONE | Face API auto-spawned. `start.bat` rewritten |
| M10 | CI/CD pipeline | ✅ DONE | `.github/workflows/ci.yml` + `test.yml` |
| M11 | README + scraper tests | ✅ DONE | |
| M12 | Remove unreliable Gemini sports facial recognition | ✅ DONE | |
| M13 | Performance optimization (zero-cost) | ✅ DONE | Parallel APIs, removed web search in image path, gzip, cache |
| M14 | Retry logic + reliability polish | ✅ DONE | `withRetry()` applied to Groq, Gemini, Jina |
| M15 | Nonescape local AI detection model | ✅ DONE | Added `/detect_ai` endpoint |
| M16 | Jersey mismatch + identity fallback | ✅ DONE | |
| M17 | ZeroGPT-style AI text detector | ✅ DONE | 7-metric ensemble at `/api/detect/text` |
| M18 | Fix Windows cp1252 Unicode crash | ✅ DONE | |
| M19 | Bump image training epochs | ✅ DONE | 1→10 |
| M20 | Fix start.bat pip install | ✅ DONE | |
| M21 | Remove invalid ESLint rule | ✅ DONE | |
| M22 | Add nonescape model download hint | ✅ DONE | |
| M23 | Add JSON 404 handler | ✅ DONE | |
| M24 | Add Python API process watchdog | ✅ DONE | Auto-restart on crash |
| M25 | Create download script for ML models | ✅ DONE | `bin/download-models.ps1` |
| M26 | Clean auto_train.py docstring | ✅ DONE | |
| M27 | Delete orphan bin/download-face.js | ✅ DONE | |
| M28 | Fix download_faces.ps1 path | ✅ DONE | |
| M29 | Fix start.bat pip check | ✅ DONE | |
| M30 | Remove invalid ESLint rule | ✅ DONE | |
| M31 | Add 404 JSON tests | ✅ DONE | |
| M32 | Fix misleading "All models trained" message | ✅ DONE | |
| M33 | Increase Face API health check retries | ✅ DONE | 3×5s → 6×7s |
| **M34** | **Switch Groq vision to direct `fetch()`** | ✅ DONE | SDK bypassed. Base64 sent directly. |
| **M35** | **Merge portrait heuristic + soften cap** | ✅ DONE | Both conditions required. Hard cap→penalty. |
| **M36** | **Remove AI Detector tab** | ✅ DONE | HTML/JS/CSS cleaned. |
| **M37** | **Fix formality weight mismatch** | ✅ DONE | Denominator matches numerator (0.25). |
| **M38** | **Quiet logs (morgan, Python, health check)** | ✅ DONE | `'short'`, filtered stderr, silent health. |
| **M39** | **Update PROJECT_MAP.md** | ✅ DONE | This file reflects current state. |
| **M40** | **Download nonescape ONNX model** | ✅ DONE | 82.7 MB downloaded from HuggingFace |
| **M41** | **Add structured async logger** | ✅ DONE | `services/logger.js` (20 lines) with ISO timestamps + 4 levels. Wired into server.js, groq.js, aiDetector.js, scraper.js. All `console.*` calls replaced project-wide. 6 new tests. |
| **M42** | **Create `.env.example`** | ✅ DONE | Template without real keys |
| **M43** | **Fix server test timeout (trainer spawn)** | ✅ DONE | Auto-trainer spawn guarded by `NODE_ENV !== 'test'` |
| **M44** | **Repair broken Nonescape ONNX + unify preprocessing** | ✅ DONE | The "ONNX" file was actually safetensors (`INVALID_PROTOBUF`), so the local image detector silently did nothing. Added `bin/convert-nonescape.py` (reconstruct `efficientnet_v2_s` → `Linear(1024,2)`, export logits), rewrote `bin/download-models.ps1`, and shared `nonescape_preprocess.py` (aspect-preserving resize → center-crop 224 → JPEG q100 → ImageNet). Runtime + eval now preprocess identically; real photos score low (≈0.00–0.05). |
| **M45** | **Leak-free text evaluation + honesty pass** | ✅ DONE | `docs/eval-welfake.md` records the dedup/group-split/calibration results (ROC-AUC 0.991, ECE 0.8%); README softened the local-model claims and linked the report. |
| **M46** | **Image evaluation harness + CIFAKE benchmark** | ✅ DONE | Added `eval/image_eval.py`, `eval/fetch_cifake.py` (+ `make_split.py`), `docs/eval-image.md`. Measured honestly: v0 is near chance on 32px CIFAKE (ROC-AUC 0.63) — a domain-shift artifact. |
| **M47** | **Nonescape demoted to advisory, then a high-confidence cap** | ✅ DONE | Removed the catastrophic `>0.65`/`>0.5` score caps (they fired on ~67–97% of real images). After the realistic benchmark, reinstated a cap **only at ≥0.9** (≈94% precision, ~3% real-image FP in both domains). `nonescapeAdvisory()` is a pure, tested function. |
| **M48** | **Realistic 512px benchmark** | ✅ DONE | The AI-vs-Human dataset reveals the model is genuinely strong on realistic images (ROC-AUC 0.89, ~80% accuracy) and only fails on low-res/OOD. Documented threshold sweep + policy in `docs/eval-image.md`. |
| **M49** | **Linear-probe image head retraining** | ✅ DONE | `bin/probe-nonescape.py` freezes the backbone, caches features, and re-fits the head (CPU-feasible). Same-domain ROC-AUC 0.89 → 0.97, ECE 0.088 → 0.018, unchanged real-photo FPs, but loses the weak cross-domain signal — so v0 stays the default and v1 is opt-in via `NONESCAPE_MODEL` (`nonescape_model.py`). |
| **M50** | **Docs truth pass** | ✅ DONE | README + this map refreshed (test counts 172 / eval 32, current line counts, resolved items, changelog). |

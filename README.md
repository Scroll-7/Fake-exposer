# AI-Powered Fake News Detector

Multi-modal deepfake detection platform: analyzes images, URLs, and text for AI-generated/manipulated content.

## Features

- **Image Analysis**: Upload or paste image URLs; detects AI generation, deepfakes, and content manipulation
- **URL Analysis**: Scrapes page content and applies both text and image analysis
- **Text Analysis**: Full Groq LLM pipeline for news/article analysis
- **AI Text Detector**: ZeroGPT-style statistical detector (no API key, runs locally)
- **Face Recognition**: InsightFace-based celebrity/player identification with jersey-mismatch detection
- **Trusted Source Boost**: +50 credibility for verified sources (NPR, Reuters, etc.)

## Architecture

```
┌─────────────┐     ┌─────────────────┐     ┌──────────────┐
│  Frontend    │────▶│  server.js       │────▶│  groq.js      │
│  (HTML/CSS)  │     │  (Express)       │     │  (LLM client) │
└─────────────┘     │                  │     └──────────────┘
                    │  services/       │     ┌──────────────┐
                    │  ├─ groq.js      │────▶│  face_api.py  │
                    │  ├─ textDetector │     │  (InsightFace) │
                    │  ├─ sportsKB.js  │     └──────────────┘
                    │  └─ scraper.js   │
                    └─────────────────┘
```

## Setup

```bash
# Install dependencies
npm install

# Environment
cp .env.example .env  # Add GROQ_API_KEY, etc.

# Start the server
npm start             # Runs on http://localhost:3000
```

The Python ML APIs (text classification, image classification, Face ID) are **not** auto-spawned by the server. Run them manually if you want those features — the server degrades gracefully without them. Requires Python 3.8+ with:
```
pip install fastapi uvicorn insightface opencv-python numpy onnxruntime

python python_api.py     # port 8000 — ML text classification
python animal_api.py     # port 8001 — ML image classification + AI detection
python face_api.py       # port 8002 — Face ID recognition
```

> **Note:** Image analysis currently accepts **JPEG and PNG only**. GIF/WebP are rejected because the vision providers (Groq vision, Gemini) cannot process them.

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/api/analyze/image` | Upload image (multipart) |
| POST | `/api/analyze/url` | Analyze URL |
| POST | `/api/analyze/text` | Analyze text content |
| POST | `/api/detect/text` | ZeroGPT-style AI text detection |
| GET | `/api/face/known` | List known face identities |

## Adding New Faces

```bash
# Using the download script
python bin/download-face.py "Person Name"

# Verify the image downloaded
ls known_faces/

# Restart the server to reload embeddings
```

The filename convention is `firstname_lastname.jpg`. The Face API converts underscores to spaces and lowercases (e.g., `elon_musk.jpg` → `elon musk`). Add an entry to `CELEBRITY_CONTEXTS` in `services/sportsKB.js` for context injection.

## Scripts

| Command | Description |
|---------|-------------|
| `npm start` | Run server |
| `npm test` | Run all tests |
| `npm run test:unit` | Run unit tests only |
| `npm run lint` | Check code style |
| `npm run lint:fix` | Auto-fix code style |
| `npm run download-face` | Download celebrity face |

## Adding to CELEBRITY_CONTEXTS

```js
'celebrity name': {
    display: 'Display Name',
    roles: ['role1', 'role2'],
    organizations: ['Org1', 'Org2'],
    typicalSettings: ['setting1', 'setting2'],
    sport: null,               // or 'basketball', 'tennis', etc.
    party: null,               // or 'Democratic', 'Republican', etc.
    opponents: [],             // political opponents if applicable
}
```

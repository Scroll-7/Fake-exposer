## **How I Built a Multi-Modal AI That Catches Deepfakes, Jersey-Swap Scams, and Text-Generated Lies**

We are drowning in misinformation. AI-generated text, deepfake images, and manipulated media are no longer coming — they're here.

A few weeks ago, a photo of Lionel Messi in a Real Madrid jersey went viral. To the untrained eye, it looked real. But Messi has never played for Real Madrid — it was a face-swap fake, the most common kind of sports misinformation.

I built **Fake Exposer** — an open-source, multi-modal fake news detection platform — to catch exactly this kind of content.

### What it does

You paste a news article, drop a URL, or upload an image. In about 20 seconds, you get back a credibility score, a verdict, red flags, green flags, and a plain-English explanation of why the system thinks the way it does.

### The architecture (the interesting part)

Most fake news detectors use a single model. This one uses a **defense-in-depth** approach with **four parallel detection signals**:

1. **Groq LLaMA** — vision analysis and fact-checking against live web search results
2. **Google Gemini** — specialized AI image forensics that checks muscle-to-skeleton ratio, skin texture consistency, lighting geometry, and edge artifacts
3. **InsightFace** — facial recognition against 32 known athletes and celebrities (catches face-swapped deepfakes)
4. **A local ONNX AI classifier** — runs entirely on-device with no API calls

Then it layers on **heuristic rules** that no LLM would think of:
- A hand-curated knowledge base of 69 soccer players across 87 clubs to catch jersey-swap fakes (hard-caps score at 8 when detected)
- A "muscle in a bedroom" detector — because real bodybuilders don't have competition-level physiques in casual mirror selfies
- An AI portrait signature detector (uniform background + centered face + studio lighting + no real-world context = 90% chance it's generated)
- A filename scanner for 15 strict AI generator keywords that trigger hard verdicts (e.g., midjourney, dall-e, dalle, stable-diffusion, flux, comfyui, nightcafe)

### The hard part

The hardest technical challenge was making this work reliably when APIs fail. Groq's free tier has tight rate limits, Gemini has quota caps, and the Python microservices may not be running. Every single layer has a fallback — if Groq vision is down, the system degrades to Gemini + heuristics. If Gemini is also down, it uses just heuristics. The server starts and runs even with zero API keys configured.

### Security was not an afterthought

The project ships with CSP headers, rate limiting (3 tiers), magic byte file validation (no trusting client MIME types), input sanitization, HTTPS redirect, file size limits, and hourly temp-file cleanup. Every image upload is validated against JPEG/PNG magic bytes before any processing, and GIF/WebP are rejected outright.

### Why I'm sharing this

Misinformation is an arms race. The people generating this content use the same AI tools we use to detect it. I believe the best defense is **transparency** — showing how detection works, sharing the code, and letting the community verify, critique, and improve.

The entire project is open-source. 99 tests pass, Playwright E2E tests included. It runs on a single Node.js process with optional Python microservices for face recognition and ML classification.

The Messi photo got a score of **8/100 — Confirmed FAKE**. That's the kind of result I want everyone to have access to.

[Link to the project]

*Thoughts? Questions? I'd love to hear from other builders working on misinformation detection.*

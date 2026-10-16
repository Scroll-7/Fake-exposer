import dotenv from 'dotenv';
import FormData from 'form-data';
import { detectAiImage, computeAiOverride } from './aiDetector.js';
import { detectExtremePhysiqueCasualSetting, detectAiPortrait, detectAiFilename } from './heuristics.js';
import { detectJerseyMismatch, findTeamInText, PLAYER_CLUBS, getCelebrityContext } from './sportsKB.js';
import { searchWeb } from './scraper.js';
import { analyzeText } from './textDetector.js';
import { logger } from './logger.js';
import { createKeyRing } from './keyring.js';
dotenv.config();

const GROQ_CHAT_URL = 'https://api.groq.com/openai/v1/chat/completions';
const GROQ_DENIED_COOLDOWN_MS = 24 * 60 * 60 * 1000;
const GROQ_RATE_LIMIT_COOLDOWN_MS = 5 * 60 * 1000;

// Failover-only Groq key pool. Provide GROQ_API_KEYS (comma-separated) for multiple
// keys; GROQ_API_KEY still works for a single key. Rebuilt when the env changes.
let groqKeyring = null;
let groqKeyringRaw = null;
function getGroqKeyring() {
    const raw = process.env.GROQ_API_KEYS || process.env.GROQ_API_KEY || '';
    if (!groqKeyring || groqKeyringRaw !== raw) {
        groqKeyring = createKeyRing(raw, { defaultCooldownMs: GROQ_RATE_LIMIT_COOLDOWN_MS });
        groqKeyringRaw = raw;
    }
    return groqKeyring;
}

export function resetGroqKeyring() {
    groqKeyring = null;
    groqKeyringRaw = null;
}

function isNetworkError(msg) {
    return /fetch failed|ECONNRESET|ETIMEDOUT|ENOTFOUND|econnrefused|network|timeout|abort/i.test(msg);
}

/**
 * Rule-based heuristic: detects if the original filename contains obvious AI generator artifacts.
 */
const ANALYSIS_SCHEMA = `
    You must respond ONLY with a valid JSON object in this exact format. Do not include markdown formatting or extra text:
    {
      "credibility_score": <number 0-100>,
      "verdict": "<string, e.g., 'Likely Fake', 'Highly Credible'>",
      "bias": "<string, e.g., 'Right-Wing', 'Center', 'None'>",
      "sentiment": "<string, e.g., 'Fear-inducing', 'Neutral'>",
      "red_flags": ["<string>", ...],
      "green_flags": ["<string>", ...],
      "summary": "<string, short explanation of verdict>",
      "recommendations": ["<string>", ...],
      "is_trusted_source": <true or false>,
      "trusted_source_name": "<string, name of the identified trusted source, or empty string if not trusted>"
    }`;

const SOURCE_INSTRUCTIONS = `
    IMPORTANT — Source Reputation Detection (Dynamic Verification):
    Try to identify WHO published or posted this content (the author, outlet, organization, or account).
    Use your visual, contextual, and internal knowledge to assess the source's credibility.

    How to determine if a source is "Highly Trusted":
    1. Visual Verification (Images): If you are analyzing a screenshot, look closely at the profile. If the account has a "verified badge" (e.g., a blue checkmark on X/Twitter, Instagram, TikTok) AND the handle/name matches a known public figure, journalist, or organization, trust it.
    2. Contextual Recognition: Recognize world-class, reputable sources across various domains. Examples include (but are not limited to):
        - Global News: BBC, Reuters, AP News, AFP, The Guardian, NYT, Washington Post, Al Jazeera, Bloomberg, WSJ.
        - Sports & Transfers: Fabrizio Romano, David Ornstein, Florian Plettenberg, Shams Charania, Adrian Wojnarowski, Sky Sports, ESPN.
        - Tech & Gaming: Marques Brownlee (MKBHD), IGN, The Verge, TechCrunch, Jason Schreier, Geoff Keighley.
        - Science & Health: WHO, CDC, NASA, Nature, Science Magazine, Neil deGrasse Tyson, Andrew Huberman.
        - Finance & Politics: Financial Times, The Economist, official government verified accounts.

    CRITICAL RULE FOR SCREENSHOTS: If the image visually contains a blue checkmark/verified badge next to the account name, you MUST set "is_trusted_source" to true. DO NOT fact-check the actual claim in the post to determine if the screenshot is a photoshop. Even if you know the claim is factually false (e.g., a fake transfer), assume the screenshot is authentic for the purpose of the "is_trusted_source" flag if the visual badge is present.`;

const QUERY_TOLERANCE_INSTRUCTIONS = `
    IMPORTANT — Casual Query Tolerance:
    Users will often submit short, casual search queries or rumors (e.g., "mbape to real madrid"). 
    DO NOT flag minor spelling mistakes (e.g., "mbape" instead of "Mbappé"), lack of capitalization, or missing punctuation as "red flags". 
    Focus entirely on the factual accuracy of the core claim, not the user's grammar. If a user is asking about a known rumor or fact, evaluate the fact itself instead of criticizing the text format.

    IMPORTANT — No Search Results Handling:
    If the search results say "could not find relevant web results", this means the web search simply failed — it does NOT mean the claim is false.
    In that case, use your general knowledge and internal reasoning to evaluate the claim.
    DO NOT mark a claim as "Likely Fake" just because no web results were found. If you cannot verify, lean toward "Unable to Verify" rather than "Likely Fake."
`;

const ML_CONFIDENCE_GATE = 0.8;

/**
 * Apply the local Python ML classifier's verdict to an analysis result.
 *
 * Pure and exported for unit testing. The local classifier is a calibrated
 * TF-IDF model trained on 2016-era political news, so it is out-of-distribution
 * for modern/sports content. It may only cap the score when genuinely confident;
 * weak predictions abstain rather than dragging real content down. Calibration is
 * measured in eval/report.py and recorded in fake_news_model.card.json.
 *
 * Mutates and returns `result`.
 */
export function applyLocalMlSignal(result, ml, gate = ML_CONFIDENCE_GATE) {
    if (!result || !ml || ml.fake_probability === undefined) return result;
    const fakeProb = Math.round(ml.fake_probability * 100);
    if (ml.is_fake && ml.fake_probability >= gate) {
        result.red_flags = result.red_flags || [];
        result.red_flags.push(`🤖 Local ML Model predicts Fake News with ${fakeProb}% probability.`);
        if (result.credibility_score > 50) {
            result.credibility_score = 50;
            result.verdict = 'Mixed / Disputed (Local ML Model flagged as Fake)';
        }
    } else if (ml.is_fake) {
        result.red_flags = result.red_flags || [];
        result.red_flags.push(`🤖 Local ML Model leans Fake (${fakeProb}%) — below the ${Math.round(gate * 100)}% confidence gate, not applied to score.`);
    } else {
        result.green_flags = result.green_flags || [];
        result.green_flags.push(`✅ Local ML Model predicts Real News with ${Math.round(ml.real_probability * 100)}% probability.`);
    }
    return result;
}

/**
 * Advisory + high-confidence cap for the local Nonescape image model.
 *
 * The model is domain-sensitive. On the realistic 512px AI-vs-human benchmark
 * it is strong (ROC-AUC 0.89; ~94% precision above 0.9), but on low-resolution
 * 32px CIFAKE it is near chance (ROC-AUC 0.63) and flags most real images as AI.
 * So:
 *   - prob > 0.9 -> `cap: true` (high confidence; ~94% precision, ~3% FP on
 *     real images in *both* benchmarks)
 *   - otherwise  -> advisory only, never caps the score
 * See docs/eval-image.md.
 *
 * Pure and exported for unit testing. Returns `{ level, note, redFlag, cap }`.
 */
export function nonescapeAdvisory(prob) {
    const empty = { level: null, note: '', redFlag: '', cap: false };
    if (prob === undefined || prob === null || Number.isNaN(prob)) return empty;
    const pct = Math.round(prob * 100);
    const weak =
        'This local model is domain-sensitive (near chance on low-resolution images), so treat it as supporting evidence only — not proof.';
    if (prob > 0.9) {
        return {
            level: 'high',
            note: `\n⚠️ LOCAL AI-IMAGE MODEL: ${pct}% AI probability (high confidence — this level is ~94% precise on real vs AI images).`,
            redFlag: `🤖 Local AI-image model: ${pct}% AI probability (high confidence)`,
            cap: true,
        };
    }
    if (prob > 0.5) {
        return {
            level: 'moderate',
            note: `\n⚠️ LOCAL AI-IMAGE MODEL (weak signal): ${pct}% AI probability. ${weak}`,
            redFlag: '',
            cap: false,
        };
    }
    if (prob < 0.3) {
        return {
            level: 'pass',
            note: `\n✅ LOCAL AI-IMAGE MODEL (weak signal): only ${pct}% AI probability — the model sees nothing suspicious.`,
            redFlag: '',
            cap: false,
        };
    }
    return empty;
}

export async function analyzeContent(text) {
    if (!getGroqKeyring().current()) {
        throw new Error('GROQ_API_KEY is not configured. Set it in .env to enable text analysis.');
    }
    // Truncate query for DuckDuckGo to avoid massive payload errors (DuckDuckGo expects short queries)
    const searchQuery = text.length > 300 ? text.substring(0, 300) : text;
    const searchResults = await searchWeb(searchQuery);

    // Short-circuit: if search couldn't find anything and text is very short,
    // skip the LLM call entirely — it wastes tokens and tends to return false
    // "Likely Fake" for unverifiable short/typo-heavy claims.
    const words = text.trim().split(/\s+/);
    if (searchResults.includes('could not find relevant web results') && words.length < 20) {
        return {
            credibility_score: 50,
            verdict: 'Unable to Verify',
            bias: 'None',
            sentiment: 'Neutral',
            red_flags: ['Text is too short for thorough fact-checking and no supporting web evidence was found. Try providing a link to a news article.'],
            green_flags: [],
            summary: 'This text is too short to fact-check reliably, and the web search could not find matching results. The claim could be true or false — more context is needed.',
            recommendations: ['Provide a link to a news article or official source', 'Provide more details about the claim'],
            is_trusted_source: false,
            trusted_source_name: ''
        };
    }

    // Truncate text for Llama model to prevent 413 Payload Too Large / Token Rate Limits
    // The TPM limit is 12000 tokens. Safely truncate to 15,000 characters.
    let contentToAnalyze = text;
    if (contentToAnalyze.length > 15000) {
        contentToAnalyze = contentToAnalyze.substring(0, 15000) + '\n...[Content truncated due to length]...';
    }

    // ZeroGPT-style statistical text detector (instant, synchronous)
    const detectorResult = analyzeText(text);
    const detectorContext = detectorResult.overallScore >= 30
        ? `\n--- STATISTICAL AI TEXT DETECTION ---
The following is an automatic statistical analysis (simulating ZeroGPT's DeepAnalyse™ methodology):
- Overall AI Probability: ${detectorResult.overallScore}%
- Perplexity (word predictability): ${detectorResult.perplexity}/100
- Burstiness (sentence length variance): ${detectorResult.burstiness}/100
- Vocabulary Diversity: ${detectorResult.vocabulary}/100
- Repetition (structural patterns): ${detectorResult.repetition}/100
- Sentence Start Diversity: ${detectorResult.sentenceStarts}/100
- Formality (AI-favored vocabulary): ${detectorResult.formality ?? '—'}/100
- AI-like sentences: ${detectorResult.aiSentencePercentage}% of all sentences
${detectorResult.overallScore >= 60 ? '\nWARNING: This text has strong statistical patterns of AI generation. Treat the content with extra skepticism.' : ''}
${detectorResult.overallScore >= 80 ? '\nCRITICAL: This text very closely matches AI writing patterns. Strong likelihood of being fully AI-generated.' : ''}
----------------------------------------`
        : '';

    const prompt = `
    You are an expert fact-checker, journalism credibility analyst, and source reputation researcher.
    Analyze the following text for signs of fake news, misinformation, bias, and manipulation.
    ${SOURCE_INSTRUCTIONS}
    ${QUERY_TOLERANCE_INSTRUCTIONS}

    --- LIVE WEB SEARCH CONTEXT (USE THIS TO FACT CHECK) ---
    The following are live search results from the internet regarding the query. 
    Use these facts to determine if the user's claim is true or false, especially for recent events:
    - ${searchResults}
    --------------------------------------------------------
    ${detectorContext}

    Text to analyze:
    "${contentToAnalyze}"

    ${ANALYSIS_SCHEMA}
    `;

    try {
        const completion = await groqTextCompletion(prompt);

        const responseText = completion.choices[0]?.message?.content || '{}';
        const result = JSON.parse(responseText);
        
        // --- LOCAL MACHINE LEARNING MODEL INTEGRATION ---
        try {
            const localApiRes = await fetch('http://127.0.0.1:8000/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ text: contentToAnalyze }),
                signal: AbortSignal.timeout(3000)
            });
            if (localApiRes.ok) {
                const localAiResult = await localApiRes.json();
                applyLocalMlSignal(result, localAiResult);
            }
        } catch {
            logger.info('Local Python ML API not reachable or timed out, skipping ML score...');
        }

        // ZeroGPT-style statistical AI text detector post-processing
        if (detectorResult.overallScore >= 60) {
            result.red_flags = result.red_flags || [];
            result.red_flags.push(`🤖 Statistical AI Detector (ZeroGPT-style): ${detectorResult.overallScore}% probability of AI-generated text (formality ${detectorResult.formality ?? '?'}%, perplexity ${detectorResult.perplexity}%, burstiness ${detectorResult.burstiness}%, ${detectorResult.aiSentencePercentage}% of sentences flagged).`);
            if (detectorResult.overallScore >= 80 && (result.credibility_score ?? 100) > 40) {
                result.credibility_score = Math.min(result.credibility_score ?? 100, 30);
                result.verdict = 'Likely AI-Generated Text';
                if (result.summary) {
                    result.summary = `Statistical AI text detection: ${detectorResult.overallScore}% AI probability. ${result.summary}`;
                }
            } else if (detectorResult.overallScore >= 60 && (result.credibility_score ?? 100) > 60) {
                result.credibility_score = Math.min(result.credibility_score ?? 100, 50);
            }
        } else if (detectorResult.overallScore > 0 && detectorResult.overallScore < 30) {
            result.green_flags = result.green_flags || [];
            result.green_flags.push(`✅ Statistical Text Detector: Only ${detectorResult.overallScore}% AI probability — text has natural human writing patterns.`);
        }

        return result;
    } catch (error) {
        logger.error('Groq text analysis error:', error);
        throw error;
    }
}

// Model catalogues are env-overridable because Groq rotates/retires model names
// per account — hardcoding them caused every request to 404 (see GROQ_TEXT_MODELS
// in .env.example). A comma-separated env value wins; an empty value means "none".
function parseModelList(envValue, defaults) {
    if (envValue === undefined) return defaults;
    return envValue.split(',').map(s => s.trim()).filter(Boolean);
}

// Text models to try in order, resolved at call time so tests/env changes apply.
const DEFAULT_TEXT_MODELS = [
    'openai/gpt-oss-120b',
    'openai/gpt-oss-20b',
    'qwen/qwen3.8-27b',
];

const DEFAULT_VISION_MODELS = [
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
];

// Try each text model in turn; a retired/unknown model falls through to the next.
// Key-level failures (401/403/429) quarantine the key and fail over to the next key.
async function groqTextCompletion(prompt) {
    const models = parseModelList(process.env.GROQ_TEXT_MODELS, DEFAULT_TEXT_MODELS);
    const keyring = getGroqKeyring();
    const keyCount = Math.max(1, keyring.size);
    let currentKey = keyring.current();
    if (!currentKey) throw new Error('GROQ_API_KEY is not configured. Set it in .env to enable text analysis.');

    let lastError;
    for (const model of models) {
        for (let keyAttempt = 0; keyAttempt < keyCount && currentKey; keyAttempt++) {
            try {
                logger.info(`Trying text model: ${model} (key attempt ${keyAttempt + 1}/${keyCount})`);
                const res = await fetch(GROQ_CHAT_URL, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${currentKey}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model,
                        messages: [{ role: 'user', content: prompt }],
                        temperature: 0.1,
                        max_tokens: 2048,
                        response_format: { type: 'json_object' },
                    }),
                    signal: AbortSignal.timeout(60000),
                });

                if (!res.ok) {
                    const errBody = await res.text().catch(() => '');
                    const msg = `${res.status}: ${errBody.slice(0, 200)}`;
                    lastError = new Error(msg);

                    // Key-level failure — quarantine and fail over to the next key.
                    if (res.status === 401 || res.status === 403 || res.status === 429) {
                        logger.warn(`Groq key rejected (HTTP ${res.status}) — quarantining key and failing over...`);
                        keyring.markBad(currentKey, res.status === 429 ? undefined : GROQ_DENIED_COOLDOWN_MS);
                        currentKey = keyring.rotate();
                        continue;
                    }

                    // Retired/unknown model — fall through to the next model.
                    if (res.status === 404 || /does not exist|do not have access|model_not_found|not found/i.test(errBody)) {
                        logger.warn(`Text model ${model} unavailable (${msg.slice(0, 120)}) — trying next...`);
                        break;
                    }

                    // Transient server error — try the next model.
                    if (res.status === 500 || res.status === 502 || res.status === 503) {
                        logger.warn(`Text model ${model} transient error (${res.status}) — trying next...`);
                        break;
                    }

                    throw new Error(`Groq API error: ${msg}`);
                }

                return await res.json();
            } catch (err) {
                const msg = err?.message || '';
                if (/^Groq API error/.test(msg)) throw err;
                if (isNetworkError(msg)) {
                    logger.warn(`Text model ${model} network error: ${msg.slice(0, 120)} — trying next...`);
                    lastError = err;
                    break;
                }
                throw err;
            }
        }
    }
    throw lastError || new Error('All text models are unavailable');
}

// Vision models to try in order — if one is over capacity, fall back to the next.
// NOTE: many Groq accounts (including this one) have no vision model at all; in that
// case analyzeImage degrades to the Gemini + heuristics path. Set GROQ_VISION_MODELS=
// (empty) to skip the futile Groq vision attempts entirely.
async function groqVisionRequest(messages, maxTokens = 500) {
    const keyring = getGroqKeyring();
    const keyCount = Math.max(1, keyring.size);
    let currentKey = keyring.current();
    if (!currentKey) throw new Error('GROQ_API_KEY is not configured');

    const models = parseModelList(process.env.GROQ_VISION_MODELS, DEFAULT_VISION_MODELS);
    let lastError;
    for (const model of models) {
        for (let keyAttempt = 0; keyAttempt < keyCount && currentKey; keyAttempt++) {
            try {
                logger.info(`Trying vision model via direct API: ${model} (key attempt ${keyAttempt + 1}/${keyCount})`);
                const res = await fetch(GROQ_CHAT_URL, {
                    method: 'POST',
                    headers: {
                        'Authorization': `Bearer ${currentKey}`,
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({
                        model,
                        messages,
                        temperature: 0.1,
                        max_tokens: maxTokens,
                    }),
                    signal: AbortSignal.timeout(60000),
                });

                if (!res.ok) {
                    const errBody = await res.text().catch(() => '');
                    const msg = `${res.status}: ${errBody.slice(0, 200)}`;

                    // Key-level failure — quarantine the key and fail over.
                    if (res.status === 401 || res.status === 403 || res.status === 429) {
                        logger.warn(`Groq vision key rejected (HTTP ${res.status}) — quarantining key and failing over...`);
                        keyring.markBad(currentKey, res.status === 429 ? undefined : GROQ_DENIED_COOLDOWN_MS);
                        currentKey = keyring.rotate();
                        lastError = new Error(msg);
                        continue;
                    }

                    const noVision = /does not support image/i.test(errBody);
                    const transient = res.status === 503 || res.status === 404 ||
                        !errBody || /over capacity|unavailable|rate limit|too many requests|temporarily/i.test(errBody);
                    if (noVision || transient) {
                        // Transient or "this model can't see images" — fall through to the next model.
                        logger.warn(`Model ${model} unavailable (${res.status}): ${msg.slice(0, 120)} — trying next...`);
                        lastError = new Error(msg);
                        break;
                    }
                    // Non-transient error (e.g. 400 invalid image data) — retrying other
                    // models would fail identically and burn quota. Abort immediately.
                    logger.warn(`Model ${model} returned non-transient error ${res.status}: ${msg.slice(0, 120)} — aborting vision request.`);
                    throw new Error(`Groq API error: ${msg}`);
                }

                const completion = await res.json();
                // Detect vision error returned as 200 OK with error text in content
                const content = completion?.choices?.[0]?.message?.content || '';
                if (/does not support image|cannot read|cannot process|unable to process/i.test(content)) {
                    logger.warn(`Model ${model} returned error text in 200 response — trying next...`);
                    lastError = new Error(content.slice(0, 200));
                    break;
                }
                logger.info(`Vision model ${model} succeeded.`);
                return completion;
            } catch (err) {
                const msg = err?.message || '';
                if (/^Groq API error/.test(msg)) throw err;
                if (isNetworkError(msg)) {
                    logger.warn(`Model ${model} network error: ${msg.slice(0, 120)} — trying next...`);
                    lastError = err;
                    break;
                }
                // Non-transient error — re-throw immediately
                throw err;
            }
        }
    }
    throw lastError || new Error('All vision models are unavailable');
}

export async function analyzeImage(imageBase64, mimeType, userContext = '', originalName = '') {
    if (!getGroqKeyring().current()) {
        throw new Error('GROQ_API_KEY is not configured. Set it in .env to enable image analysis.');
    }
    // Groq enforces a 4 MB limit on base64-encoded images
    const base64SizeBytes = Math.ceil(imageBase64.length * 3 / 4);
    const MAX_BASE64_BYTES = 4 * 1024 * 1024; // 4 MB
    if (base64SizeBytes > MAX_BASE64_BYTES) {
        throw new Error(`Image too large (${(base64SizeBytes / (1024 * 1024)).toFixed(1)} MB). Maximum is 4 MB. Please resize or compress the image.`);
    }

    const dataUrl = `data:${mimeType};base64,${imageBase64}`;

    // Build filename context note if the name looks suspicious (instant, no async)
    const filenameNote = originalName
        ? `\n\nFILE METADATA: The original filename is "${originalName}". If this name contains references to AI tools (chatgpt, midjourney, dalle, stable diffusion, flux, etc.) or words like "fake", "edited", "enhanced", "generated" — treat it as a STRONG signal that the image is AI-generated or manipulated and score accordingly.`
        : '';

    // ── PARALLEL: Face API + Gemini + Nonescape + Groq Step 1 ──
    // All are independent — run concurrently to cut ~10s off the total time.
    const [faceIdOverride, aiDetection, nonescapeResult, imageDescription] = await Promise.all([

        // Face API (Python microservice)
        (async () => {
            try {
                logger.info('[FaceID] Sending image to local Python Face API...');
                const buffer = Buffer.from(imageBase64, 'base64');
                const form = new FormData();
                form.append('file', buffer, { filename: 'upload.jpg', contentType: mimeType });
                const faceRes = await fetch('http://127.0.0.1:8002/recognize', {
                    method: 'POST', body: form, headers: form.getHeaders(),
                    signal: AbortSignal.timeout(10000)
                });
                if (faceRes.ok) {
                    const faceData = await faceRes.json();
                    if (faceData && faceData.player) {
                        logger.info(`[FaceID] MATCH FOUND! The person is exactly: ${faceData.player}`);
                        return faceData.player;
                    }
                    logger.info(`[FaceID] No match found in known_faces dataset: ${faceData.reason || 'None'}`);
                } else {
                    logger.warn('[FaceID] API returned error:', faceRes.status);
                }
            } catch (e) {
                logger.warn('[FaceID] Failed to call Python Face API:', e.message);
            }
            return null;
        })(),

        // Gemini AI image forensics
        detectAiImage(imageBase64, mimeType),

        // Nonescape local ONNX AI detection model (port 8001)
        (async () => {
            try {
                const buffer = Buffer.from(imageBase64, 'base64');
                const form = new FormData();
                form.append('file', buffer, { filename: 'image.jpg', contentType: mimeType });
                const res = await fetch('http://127.0.0.1:8001/detect_ai', {
                    method: 'POST', body: form, headers: form.getHeaders(),
                    signal: AbortSignal.timeout(10000)
                });
                if (res.ok) return await res.json();
            } catch (e) {
                logger.warn('[Nonescape] AI detection API not reachable:', e.message);
            }
            return null;
        })(),

        // Step 1: Rich visual description
        (async () => {
            logger.info('Starting Groq step 1 (Description)...');
            const descStartTime = Date.now();
            try {
                const userContextNote = userContext
                    ? `\n\nIMPORTANT — The user says this image: "${userContext}". Keep this in mind.`
                    : '';
                const descCompletion = await groqVisionRequest([
                    {
                        role: 'user',
                        content: [
                            {
                                type: 'text',
                                text: `You are a visual forensics expert. Analyze this image thoroughly and answer ALL of the following questions:${userContextNote}${filenameNote}

--- GENERAL AI / MANIPULATION DETECTION ---
1. Does this look AI-generated or AI-edited? Look for:
   - Unnaturally smooth or waxy skin texture (Note: account for bodybuilding stage oil/spray tan which looks waxy)
   - Exaggerated or impossible muscle definition / body proportions (Note: extreme conditioning in professional bodybuilders is real, consider context)
   - Inconsistent lighting between body parts (e.g. face lit differently from torso)
   - Blurring or smearing at body edges / background seams
   - Unnatural background consistency or bokeh patterns
   - Finger/hand deformities or repetition
   - Skin color or tone mismatches between face and body
2. Does the body look naturally proportioned, or are features (muscles, face, limbs) exaggerated beyond what is humanly typical for the person's frame?
3. Are there any blending artifacts or unnatural boundary transitions between body parts or between the person and the background?

--- IDENTITY CHECK (SPORTS IMAGES ONLY) ---
4. If the person is wearing a sports jersey: identify the TEAM from the badge, sponsor text, or jersey colors/pattern. For the player: if their NAME is printed on the jersey (back or front), use that. If no name is visible, you MUST name the player if they are an ultra-famous global superstar whose face you clearly recognise (e.g. Messi, Ronaldo, Neymar, Mbappé, Salah, Lewandowski, De Bruyne, Mbappé). This is CRITICAL for detecting fake/photoshopped images where a star's head is placed on another player's body. For all other players, say "Player name not visible in image".

--- SUMMARY ---
6. Write a 2-3 sentence summary of what the image shows and whether it appears authentic or manipulated.

Be very specific and honest about uncertainty.`
                            },
                            { type: 'image_url', image_url: { url: dataUrl } }
                        ]
                    }
                ], 250);
                let desc = descCompletion.choices[0]?.message?.content || '';
                // If the model returns an error instead of a description, treat as failed
                if (/does not support image|cannot read|cannot process|model.*not support|unable to process/i.test(desc)) {
                    logger.warn(`Groq Description returned error text instead of description: "${desc.slice(0, 120)}"`);
                    desc = '';
                }
                logger.info(`Groq Description completed in ${Date.now() - descStartTime}ms. Output length: ${desc.length}`);
                return desc;
            } catch (err) {
                logger.error(`Groq Description failed after ${Date.now() - descStartTime}ms:`, err.message);
                return '';
            }
        })(),
    ]);

    // Compute AI override from Gemini result (instant, local)
    const aiOverride = computeAiOverride(aiDetection);

    // Nonescape local ONNX model result — advisory only (see nonescapeAdvisory).
    const nonescapeAiProb = nonescapeResult?.ai_probability;
    const nonescapeInfo = nonescapeAdvisory(nonescapeAiProb);
    const nonescapeNote = nonescapeInfo.note;

    // Build face ID context for Step 3 (no longer modifying userContext in-place)
    const faceIdContext = faceIdOverride
        ? `[GROUND TRUTH FACE ID] The face in this image mathematically matches ${faceIdOverride}. DO NOT guess the name, it is absolutely ${faceIdOverride}.`
        : '';

    // ── CELEBRITY CONTEXT ──
    // If the Face API identified a known celebrity, add their known affiliations/roles
    // so the LLM can detect out-of-context or implausible depictions.
    const celebrityInfo = getCelebrityContext(faceIdOverride);
    const celebrityNote = celebrityInfo
        ? `\n[CELEBRITY PROFILE] ${celebrityInfo.display} is known as a ${celebrityInfo.roles.join(', ')}. ` +
          (celebrityInfo.organizations.length ? `Associated with: ${celebrityInfo.organizations.join(', ')}. ` : '') +
          `Typical settings: ${celebrityInfo.typicalSettings.join(', ')}. ` +
          (celebrityInfo.party ? `Political party: ${celebrityInfo.party}. ` : '') +
          (celebrityInfo.opponents?.length ? `Political opponents: ${celebrityInfo.opponents.join(', ')}. ` : '') +
          `If this image depicts ${celebrityInfo.display} in an unusual or implausible setting or alongside political opponents that contradicts their known profile, this is a STRONG signal of a deepfake or manipulation.`
        : '';

    // Note: searchWeb() is intentionally omitted from the image analysis path.
    // Web search is designed for text fact-checking — for images it adds 5-10s
    // of unreliable DuckDuckGo scraping with negligible factual value.
    const searchResults = 'Search engines could not find relevant web results for this query. This does not mean the claim is false — only that no matching pages were found online. Fact-check based on general knowledge and internal reasoning.';

    // ── STEP 1.5a: Local KB jersey mismatch check (offline, instant) ──
    // Check the Step 1 description, user context, and Face API result combined.
    // The LLM is instructed NOT to guess the player from facial features, so it may
    // avoid naming the player even when the face is obvious. The user's own input
    // ("Is Neymar in an AC Milan jersey real?") fills that gap.
    const mismatchText = [imageDescription || aiDetection?.description, userContext, faceIdOverride, aiDetection?.reasoning].filter(Boolean).join(' ');
    let jerseyMismatch = detectJerseyMismatch(mismatchText);
    // Clear jersey mismatch if text contains transfer-news keywords — the check is
    // for PHOTOS of players in the wrong jersey, not for text articles about transfers
    // Only check user-provided text for transfer keywords — AI-generated analysis
    // text commonly contains false-positives like "transfer" (face transfer), "deal" (digital manipulation).
    if (jerseyMismatch && /signs?|joins?|transfers?|announces?|agreement|deal|completed|confirmed|here we go|move to|has joined|has signed/i.test(`${userContext} ${faceIdOverride || ''}`)) {
        jerseyMismatch = null;
    }

    // Fallback: if a team jersey IS clearly visible but NO player name can be
    // matched from any source (description, user context, or Face API), flag
    // this as suspicious — the person's identity is unverifiable.
    let unverifiableIdentity = null;
    if (!jerseyMismatch) {
        const teamName = findTeamInText(mismatchText);
        if (teamName && !Object.keys(PLAYER_CLUBS).some(p => mismatchText.toLowerCase().includes(p))) {
            unverifiableIdentity = teamName;
            logger.warn(`Unverifiable identity: jersey shows ${teamName} but no player name found in text`);
        }
    }

    // ── STEP 1.5b: Sports context suspicion (when Face API is unavailable) ──
    // If we couldn't run the Face API (faceIdOverride is null) AND the image
    // description suggests sports content but no player+team combo was matched,
    // add a suspicion flag. This catches cases where the LLM describes a player
    // in a jersey but doesn't name them (e.g., Messi in a Real Madrid jersey
    // described as "a man in a white jersey").

    // ── STEP 2.5: Rule-based heuristics ──
    // When Groq vision fails (empty description), use Gemini's description as fallback
    const heuristicDesc = imageDescription || aiDetection?.description || aiDetection?.reasoning || '';
    const sportsWarning = detectSportsSuspicion(faceIdOverride, jerseyMismatch, heuristicDesc, findTeamInText, aiDetection?.imageCategory);
    const physiqueWarning = detectExtremePhysiqueCasualSetting(heuristicDesc);
    if (physiqueWarning) {
        logger.warn(`Physique heuristic triggered: ${physiqueWarning}`);
    }

    const portraitWarning = detectAiPortrait(heuristicDesc);
    if (portraitWarning) {
        logger.warn(`Portrait heuristic triggered: ${portraitWarning}`);
    }

    const filenameWarning = detectAiFilename(originalName);
    if (filenameWarning) {
        logger.warn(`Filename heuristic triggered: ${filenameWarning}`);
    }

    const anyHeuristicWarning = physiqueWarning || portraitWarning || filenameWarning;

    // ── UNVERIFIABLE IDENTITY FALLBACK ──
    // If a team jersey was detected but no player name could be matched,
    // add a caution flag for the Step 3 LLM.
    const unverifiableNote = unverifiableIdentity
        ? `\n⚠️ AUTOMATIC IDENTITY WARNING: The jersey in this image shows ${unverifiableIdentity}, but the system could not determine who the person is — no player name is visible on the jersey, the user did not name them, and Face ID did not match. This combination (clear team jersey + unidentifiable person) is suspicious and may indicate a face-swapped fake where a star's head was placed on another player's body. Treat this as a STRONG signal for low credibility (score < 40) unless you can clearly identify the person\'s face.`
        : '';

    // ── SPORTS CONTEXT SUSPICION ──
    const sportsNote = sportsWarning
        ? `\n⚠️ AUTOMATIC SPORTS SUSPICION: ${sportsWarning}\nThis appears to be a sports-related image but the system could not identify the specific team or player. Without the Face API available for identity verification, this image should be treated with moderate suspicion.`
        : '';

    // ── STEP 3: Full analysis with both visual + factual context ──
    const userContextSection = userContext || filenameNote || faceIdContext || anyHeuristicWarning || unverifiableIdentity || nonescapeNote || sportsWarning || celebrityNote
        ? `--- USER-PROVIDED CONTEXT (TREAT AS A STRONG SIGNAL) ---
    ${userContext ? `The person who uploaded this image says: "${userContext}"` : ''}
    ${faceIdContext ? `\n${faceIdContext}` : ''}
    ${celebrityNote}
    ${filenameNote}
    ${physiqueWarning ? `\n⚠️ AUTOMATIC HEURISTIC WARNING (MUSCLE): ${physiqueWarning}\nThis combination (extreme competition-level physique + casual everyday setting) is a PRIMARY indicator of AI muscle enhancement. You MUST reflect this suspicion in your score and verdict. Score should be 30 or lower.` : ''}
    ${portraitWarning ? `\n⚠️ AUTOMATIC HEURISTIC WARNING (PORTRAIT): ${portraitWarning}\nThis image matches the signature of AI-generated portrait photos from tools like Gemini Image, DALL-E, and Midjourney. You MUST reflect this suspicion. Score should be 25 or lower unless you find specific real-world evidence this is genuine.` : ''}
    ${filenameWarning ? `\n⚠️ AUTOMATIC HEURISTIC WARNING (FILENAME): ${filenameWarning}\nYou MUST score this as extremely low credibility (1-10) because the filename itself reveals it is an AI generation.` : ''}
    ${unverifiableNote}
    ${nonescapeNote}
    ${sportsNote}
    If the user admits the image is AI-generated, edited, or fake, TRUST THEM and reflect this in your verdict and score.
    If the filename suggests AI origin, treat it as a high-confidence signal of manipulation.
    --------------------------------------------------------`
        : '';

    const prompt = `
    You are an expert fact-checker, journalism credibility analyst, visual forensics expert, and source reputation researcher.
    You are given an image (screenshot, photo, or document).

    ${userContextSection}

    STEP 1: Determine the image type:
    - Type A: Contains readable text (social media post, news headline, article, caption)
    - Type B: Primarily a photograph or graphic without significant text

    STEP 2A — If TYPE A (text-based):
      Extract all visible text and analyze it for fake news, misinformation, bias, and manipulation.
      ${SOURCE_INSTRUCTIONS}
      ${QUERY_TOLERANCE_INSTRUCTIONS}

    STEP 2B — If TYPE B (photo/graphic):
      You MUST perform ALL of the following checks:

      ── CLASSIFY THE PHOTO TYPE FIRST ──
      Is this:
      (a) A professional bodybuilding COMPETITION photo (stage, spotlight, posing trunks, audience, spray tan, banner)?
      (b) A casual mirror selfie / gym selfie / bathroom selfie (person holding phone in mirror, everyday setting)?
      (c) A portrait, headshot, or close-up (person facing camera, mostly face/shoulders visible, often against a plain or simple background)?
      (d) A group photo / crowd scene (multiple people, events, parties, background characters)?
      (e) A politician, celebrity, or public figure in a dramatic or unusual situation?
      (f) A sports action shot or other?
      Your strictness level depends on this classification.

      ── FOR CASUAL MIRROR SELFIES (category b) — MANDATORY AI MUSCLE ENHANCEMENT CHECK ──
      This is the most common type of fake image. Go through each check:

      CHECK A — MUSCLE-TO-SKELETON RATIO (most important):
        → Look at thin bones: wrists, forearms, collarbone, neck thickness, jaw width.
        → Now compare to muscle volume: arms, chest, shoulders, abs definition.
        → If muscles look disproportionately large, defined, or "pumped" relative to the visible
          skeletal frame, this is a PRIMARY sign of AI enhancement.
        → A real lean physique will have proportional relationship between bone size and muscle size.
          Extremely wide, thick muscles on someone with average-sized wrists/collar = AI red flag.

      CHECK B — SKIN TEXTURE FACE VS BODY:
        → Compare skin on the FACE (forehead, cheeks, nose) vs TORSO (chest, abs, arms).
        → Natural photo: same grain, pores, and texture variation everywhere.
        → AI-edited: torso skin often looks smoother, cleaner, more "airbrushed" than the face.
        → If the torso skin looks noticeably more rendered or plastic than the face, flag it.

      CHECK C — LIGHTING GEOMETRY:
        → Identify the key light direction from face shadows.
        → Do muscle highlights and shadows on the torso follow the SAME light direction?
        → AI muscle edits often have body shading that doesn't match the ambient room lighting.

      CHECK D — EDGE ARTIFACTS AT BODY SILHOUETTE:
        → Look at the contour edges where arms/shoulders/torso meet the background.
        → Are there halos, blurring, unnatural sharpening, or a subtle "glow" at muscle edges?
        → AI body size increases often leave artifacts at the outline.

      CHECK E — SKIN TONE MISMATCH:
        → Is the torso skin slightly more saturated, more tanned, or more reddish than the face?
        → Natural selfies: consistent skin tone everywhere.
        → AI edits often subtly shift the body color.

      CHECK F — BODYBUILDER CONTEXT:
        → Is this person CLEARLY a professional competitive bodybuilder (stage, trunks, competition banner, extreme conditioning with stage tan)?
        → If YES → be lenient. Elite competitive bodybuilders genuinely have extreme physiques.
        → If NO (casual selfie, bathroom, bedroom, normal clothes) → be STRICT.
        → A person in a casual mirror selfie with a physique that looks like a professional stage-ready bodybuilder is extremely suspicious.

      ── FOR STUDIO PORTRAITS (category c) — MANDATORY AI PORTRAIT CHECK ──
      AI image generators (like Midjourney, DALL-E, Gemini) are heavily biased toward generating portraits with these specific flaws:
      → Perfectly uniform grey, white, or neutral background without real-world depth or clutter.
      → "Studio" lighting that is perfectly symmetrical and flawless.
      → Skin texture that lacks real-world blemishes, asymmetric pores, or peach fuzz (often looks hyper-real or plastic).
      → Eyes that have mismatched catchlights (reflections) or perfectly circular irises.
      → Lack of real-world environmental context.
      If it looks like a "perfect passport photo" without any real-world messiness, flag it heavily.

      ── FOR GROUP PHOTOS & CROWDS (category d) — MANDATORY AI CROWD CHECK ──
      → Look closely at the faces of people in the background. AI models often generate mangled, melting, or featureless faces for background characters.
      → Count fingers and look at hands. Are hands merging into other people's bodies or clothing? Are there extra limbs?
      → Look at background text or signs. Are they written in a nonsensical alien language (common AI artifact)?

      ── FOR PUBLIC FIGURES (category e) — MANDATORY AI DEEPFAKE CHECK ──
      → Does the public figure look overly glossy, dramatized, or caricatured compared to real press photos?
      → AI often generates politicians with exaggerated expressions, perfect cinematic lighting, or six-fingered hands.
      → IF this is a public figure in an unusual situation (e.g. being arrested, wearing a puffy jacket, doing something scandalous), it is almost certainly an AI deepfake.

      ── SCORING FOR CASUAL SELFIES, PORTRAITS, GROUPS ──
      • Clear AI artifacts (muscle mismatch, plastic skin, uncanny uniform background, mangled background faces, text anomalies) → LOW (5-25), verdict "Likely AI-Generated / Edited"
      • Suspicious but minor (minor skin smoothing, uncertain lighting) → MEDIUM-LOW (25-40), verdict "Suspicious — Possible AI Enhancement"
      • Authentic-looking photo, real-world imperfections and messy background → HIGH (70-90)
      • Professional competition photo with no obvious AI artifacts → HIGH (75-95)

      ── GENERAL SCORING (all photo types) ──
      • No AI artifacts + factually accurate → HIGH (80-100)
      • Subtle AI artifacts OR minor factual uncertainty → MEDIUM (35-65)
      • Clear AI artifacts (muscle/body enhancement, skin inconsistencies, etc.) → LOW (5-30), verdict "Likely AI-Generated / Edited"
      • User admitted it is AI-generated or fake → VERY LOW (5-15), verdict "Confirmed Fake / AI-Generated"
      • Wrong sports jersey for identified player → VERY LOW (5-20), verdict "Fake / Manipulated"

      ── IDENTITY CAUTION RULE ──
      - If the person is NOT a well-known public figure you are 80%+ confident about, say "unidentified person".
      - Similarity to a celebrity is NOT identification.

      ── SPORTS JERSEY FACT-CHECK (only if applicable) ──
      Identify the TEAM from the jersey badge/colors/sponsor text.
      Identify the PLAYER:
        - If a name IS visible on the jersey (front or back), use that.
        - If NO name is visible but the face is of an ultra-famous global superstar (e.g. Messi, Ronaldo, Neymar, Mbappé, Salah, Lewandowski, De Bruyne) whose face you clearly recognise, you MUST name them for mismatch detection. This is critical for catching face-swapped fakes.
        - Otherwise say "Player name not visible".
      If the identified player has NEVER played for the team on the jersey → score 5-20, verdict "Fake / Manipulated - Player Not at This Club"

    --- VISUAL ANALYSIS FROM STEP 1 ---
    ${imageDescription || 'No description available.'}

    --- LIVE WEB SEARCH CONTEXT ---
    ${searchResults}
    --------------------------------------------------------

    ${ANALYSIS_SCHEMA}
    `;

    try {
        logger.info('Starting Groq step 3 (Final Analysis)...');
        const analysisStartTime = Date.now();
        const completion = await groqVisionRequest([
            {
                role: 'user',
                content: [
                    { type: 'text', text: prompt },
                    { type: 'image_url', image_url: { url: dataUrl } }
                ]
            }
        ], 1024);
        logger.info(`Groq Analysis completed in ${Date.now() - analysisStartTime}ms.`);

        let responseText = completion.choices[0]?.message?.content || '{}';
        // Clean markdown blocks if vision model ignores json_object format
        responseText = responseText.replace(/```json/g, '').replace(/```/g, '').trim();

        // If the model returned an error message instead of JSON, treat as vision-down
        if (/does not support image|cannot read|cannot process|model.*not support|unable to process/i.test(responseText)) {
            throw new Error(`Vision model returned error: ${responseText.slice(0, 200)}`);
        }

        let llmResult;
        try {
            llmResult = JSON.parse(responseText);
        } catch (parseErr) {
            throw new Error(`Vision response was not valid JSON. ${parseErr.message}`, { cause: parseErr });
        }
        llmResult.face_identified = faceIdOverride || null;

        // ── POST-STEP-3 RE-CHECK FOR JERSEY MISMATCH ──
        // The LLM might name the player/team in its verdict even when Step 1 didn't.
        // Also includes user context and Face API result (same as the pre-check).
        // Skip for transfer announcements, screenshots, or trusted sources.
        if (!jerseyMismatch) {
            const combinedLlmText = `${llmResult.summary || ''} ${llmResult.verdict || ''} ${(llmResult.red_flags || []).join(' ')} ${userContext || ''} ${faceIdOverride || ''}`;
            // Only check user-provided text for transfer keywords (see pre-check comment above)
            const isLLMTransfer = /signs?|joins?|transfers?|announces?|agreement|deal|completed|confirmed|here we go|move to|has joined|has signed/i.test(`${userContext} ${faceIdOverride || ''}`);
            const llmHasSource = extractTrustedSource(combinedLlmText);
            if (!isLLMTransfer && !llmHasSource && aiDetection?.imageCategory !== 'Screenshot') {
                const postCheckMismatch = detectJerseyMismatch(combinedLlmText);
                if (postCheckMismatch) {
                    logger.info(`[Sports Check] Post-LLM check caught mismatch: ${postCheckMismatch.player} + ${postCheckMismatch.jerseyTeam}`);
                    jerseyMismatch = postCheckMismatch;
                }
            }
        }

        // ── MERGE AI OVERRIDE: Apply Gemini detection result ──
        if (aiOverride) {
            const artifactFlags = (aiOverride.ai_artifacts || []).map(a => `⚠️ AI artifact: ${a}`);
            const aiFlag = `🤖 Gemini AI Detector: ${aiOverride.ai_probability}% probability of AI generation/editing`;

            // SOFT WARNING: suspicious but below hard-override threshold
            // Inject red flags + modestly reduce score, but keep Groq's verdict
            if (aiOverride.softWarning) {
                const originalScore = llmResult.credibility_score ?? 80;
                // Reduce score proportionally: higher AI probability = bigger penalty
                const penalty = Math.round(aiOverride.ai_probability * 0.6);
                const adjustedScore = Math.max(5, originalScore - penalty);
                logger.info(`Soft AI warning: original score ${originalScore} → adjusted ${adjustedScore} (Gemini: ${aiOverride.ai_probability}%)`);
                return {
                    ...llmResult,
                    credibility_score: adjustedScore,
                    verdict: adjustedScore < 40
                        ? `Suspicious — Possible AI Enhancement (${aiOverride.ai_probability}% AI signal)`
                        : llmResult.verdict,
                    red_flags: [
                        ...artifactFlags,
                        ...(llmResult.red_flags || []),
                        aiFlag,
                    ],
                    green_flags: adjustedScore < 40 ? [] : (llmResult.green_flags || []),
                    summary: aiOverride.ai_reasoning
                        ? `${aiOverride.ai_reasoning} ${llmResult.summary || ''}`
                        : llmResult.summary,
                    ai_detection: {
                        probability: aiOverride.ai_probability,
                        verdict: aiDetection?.verdict,
                        confidence: aiDetection?.confidence,
                        artifacts: aiOverride.ai_artifacts,
                    }
                };
            }

            // HARD OVERRIDE: definite or likely AI
            return {
                ...llmResult,
                credibility_score: aiOverride.credibility_score,
                verdict: aiOverride.verdict,
                is_trusted_source: false, // Forcibly remove trusted boost for FAKES
                trusted_source_name: null,
                red_flags: [
                    ...artifactFlags,
                    ...(llmResult.red_flags || []),
                    aiFlag,
                ],
                green_flags: [],  // Clear green flags — the image is fake
                summary: aiOverride.ai_reasoning || llmResult.summary,
                ai_detection: {
                    probability: aiOverride.ai_probability,
                    verdict: aiDetection?.verdict,
                    confidence: aiDetection?.confidence,
                    artifacts: aiOverride.ai_artifacts,
                }
            };
        }

        // ── JERSEY MISMATCH CAP (highest priority after AI override) ──
        // If the local KB confirmed a player/team mismatch, hard-cap the score.

        if (jerseyMismatch) {
            const score = 8;
            logger.warn(`Jersey mismatch hard cap: ${jerseyMismatch.player} in ${jerseyMismatch.jerseyTeam} jersey — score → ${score}`);
            return {
                ...llmResult,
                credibility_score: score,
                verdict: `Fake / Manipulated — ${jerseyMismatch.player} never played for ${jerseyMismatch.jerseyTeam}`,
                is_trusted_source: false, // Forcibly remove trusted boost for FAKES
                trusted_source_name: null,
                red_flags: [
                    `⚽ Jersey mismatch: ${jerseyMismatch.player} has NEVER played for ${jerseyMismatch.jerseyTeam}`,
                    `📋 Known clubs: ${jerseyMismatch.knownClubs}`,
                    'This is a classic fake sports photo — a real player photograph with a digitally swapped jersey',
                    ...(llmResult.red_flags || []),
                ],
                green_flags: [],
                summary: `${jerseyMismatch.mismatchMsg} ${llmResult.summary || ''}`.trim(),
            };
        }

        // ── HEURISTIC SCORE CAP ──
        // If either heuristic fired and Groq still returned a high score, cap it.
        // Groq often ignores prompt warnings and returns high scores anyway.

        if (portraitWarning && (llmResult.credibility_score ?? 100) > 30) {
            const originalScore = llmResult.credibility_score;
            const penalty = Math.min(40, Math.round(originalScore * 0.55));
            const adjustedScore = Math.max(5, originalScore - penalty);
            logger.warn(`Portrait heuristic penalty: score ${originalScore} → ${adjustedScore} (penalty: ${penalty})`);
            return {
                ...llmResult,
                credibility_score: adjustedScore,
                verdict: adjustedScore < 30
                    ? 'Suspicious — Possible AI Portrait'
                    : llmResult.verdict,
                red_flags: [
                    '🤖 AI portrait signature detected: uniform/neutral background + centered face + studio lighting + no real-world context',
                    'This exact composition is the default output of AI image generators (Gemini Image, DALL-E, Midjourney, Stable Diffusion)',
                    ...(llmResult.red_flags || []),
                ],
                green_flags: adjustedScore < 30 ? [] : (llmResult.green_flags || []),
                summary: `Heuristic note: ${portraitWarning} ${llmResult.summary || ''}`.trim(),
            };
        }

        if (physiqueWarning && (llmResult.credibility_score ?? 100) > 35) {
            const cappedScore = 20;
            logger.warn(`Physique heuristic cap applied: score ${llmResult.credibility_score} → ${cappedScore}`);
            return {
                ...llmResult,
                credibility_score: cappedScore,
                verdict: 'Likely AI-Enhanced / Suspicious',
                red_flags: [
                    '🏋️ Extreme competition-level physique detected in a casual home/mirror setting',
                    'This specific combination (extreme muscle definition + everyday bathroom/bedroom setting) is a primary indicator of AI body enhancement',
                    ...(llmResult.red_flags || []),
                ],
                green_flags: [],
                summary: `Automatic heuristic flagged this image: ${physiqueWarning} ${llmResult.summary || ''}`.trim(),
            };
        }

        if (filenameWarning && (llmResult.credibility_score ?? 100) > 10) {
            const cappedScore = 5;
            logger.warn(`Filename heuristic cap applied: score ${llmResult.credibility_score} → ${cappedScore}`);
            return {
                ...llmResult,
                credibility_score: cappedScore,
                verdict: 'Confirmed AI-Generated',
                red_flags: [
                    '🤖 The file\'s original name explicitly reveals it was created by an AI tool',
                    ...(llmResult.red_flags || []),
                ],
                green_flags: [],
                summary: `Automatic heuristic: ${filenameWarning} ${llmResult.summary || ''}`.trim(),
            };
        }

        if (unverifiableIdentity && (llmResult.credibility_score ?? 100) > 45) {
            const cappedScore = 15;
            logger.warn(`Unverifiable identity cap: jersey shows ${unverifiableIdentity} but no player name — score ${llmResult.credibility_score} → ${cappedScore}`);
            return {
                ...llmResult,
                credibility_score: cappedScore,
                verdict: 'Suspicious — Unverifiable Person in Team Jersey',
                red_flags: [
                    `⚽ The jersey clearly shows ${unverifiableIdentity}, but who the person is could not be determined — no visible player name, no face match, and the user didn't identify them`,
                    'This combination (clear team brand + unidentifiable person) often indicates a face-swapped fake',
                    ...(llmResult.red_flags || []),
                ],
                green_flags: [],
                summary: `Identity unverifiable: ${unverifiableIdentity} jersey with no identifiable player. ${llmResult.summary || ''}`.trim(),
            };
        }

        // ── SPORTS CONTEXT SUSPICION CAP ──
        // When Face API is unavailable and sports content is detected but neither
        // team nor player could be identified, cap at 60 (moderate suspicion)
        // to avoid 100% "definitely real" for unverifiable sports images.
        if (sportsWarning && (llmResult.credibility_score ?? 100) > 70) {
            const cappedScore = 55;
            logger.warn(`Sports context suspicion cap: sports content without identifiable team/player — score ${llmResult.credibility_score} → ${cappedScore}`);
            return {
                ...llmResult,
                credibility_score: cappedScore,
                verdict: 'Unverifiable — Sports Image Without Identifiable Details',
                red_flags: [
                    '⚽ Sports image detected but the player or team could not be identified. Unverifiable sports images may be altered.',
                    ...(llmResult.red_flags || []),
                ],
                summary: `Sports context detected without verifiable team or player identity. ${llmResult.summary || ''}`.trim(),
            };
        }

        // ── NONESCAPE HIGH-CONFIDENCE AI CAP (>= 0.9) ──
        // Only a very high probability caps the score. The model is
        // domain-sensitive and unreliable below this (see nonescapeAdvisory and
        // docs/eval-image.md), but at >= 0.9 it is ~94% precise with ~3% false
        // positives on real images across two independent benchmarks.
        if (nonescapeInfo.cap && (llmResult.credibility_score ?? 100) > 40) {
            const pct = Math.round(nonescapeAiProb * 100);
            logger.warn(`Nonescape high-confidence AI cap: ${pct}% — score ${llmResult.credibility_score} -> 30`);
            return {
                ...llmResult,
                credibility_score: 30,
                verdict: 'Likely AI-Generated (ML Detector)',
                red_flags: [
                    `🤖 Local AI-image model: ${pct}% AI probability (high confidence)`,
                    ...(llmResult.red_flags || []),
                ],
                green_flags: [],
                summary: `AI-image model flagged this image with high confidence (${pct}%). ${llmResult.summary || ''}`.trim(),
            };
        }

        return llmResult;
    } catch (error) {
        const detail = error?.error?.message || error?.message || 'Unknown error';

        // ── FALLBACK: When Groq vision is unavailable, return a result from
        // Gemini + heuristics + local models instead of crashing.
        const isVisionDown = /vision model|does not support image|cannot read|cannot process|unable to process|model.*not support|unavailable|rate limit|429|does not exist|do not have access|model_not_found|not found|404|decommission|deprecat/i.test(detail);
        if (isVisionDown) {
            logger.warn('Groq vision unavailable — falling back to Gemini + local models.');
        } else {
            logger.error('Groq image analysis error:', detail, error);
        }

        // Safety net: if the fallback path itself throws, return a safe default
        try {

        if (isVisionDown) {

            // Build fallback result from available data
            const fallback = {
                credibility_score: 50,
                verdict: 'Unable to Complete Full Analysis (Vision Model Unavailable)',
                bias: 'Unknown',
                sentiment: 'Neutral',
                red_flags: [],
                green_flags: [],
                summary: 'Groq vision analysis was unavailable. Results are based on local detection models only.',
                recommendations: ['Try again later when the vision API is available.', 'Consider using Text or URL analysis as an alternative.'],
                is_trusted_source: false,
                trusted_source_name: '',
                face_identified: faceIdOverride || null,
            };

            // Screenshots don't trigger portrait/physique/sports heuristics
            const isScreenshot = aiDetection?.imageCategory === 'Screenshot';

            // Inject heuristics as red flags (skip all for screenshots)
            if (!isScreenshot && portraitWarning) {
                fallback.red_flags.push(`🤖 ${portraitWarning}`);
                fallback.credibility_score = Math.min(fallback.credibility_score, 35);
                fallback.summary = `Local heuristic flagged this as a possible AI portrait. ${fallback.summary}`;
            }
            if (!isScreenshot && physiqueWarning) {
                fallback.red_flags.push(`🏋️ ${physiqueWarning}`);
                fallback.credibility_score = Math.min(fallback.credibility_score, 30);
            }
            if (filenameWarning) {
                fallback.red_flags.push(`📁 ${filenameWarning}`);
                fallback.credibility_score = 5;
                fallback.verdict = 'Confirmed AI-Generated (Filename Reveals AI Origin)';
            }

            // Check if Gemini description mentions a known trusted source
            const combinedText = [userContext, aiDetection?.description, aiDetection?.reasoning].filter(Boolean).join(' ');
            const foundSource = extractTrustedSource(combinedText);
            if (foundSource) {
                fallback.is_trusted_source = true;
                fallback.trusted_source_name = foundSource;
            }

            // Detect transfer-news keywords — if the text says "signs", "joins", "here we go"
            // etc., the jersey mismatch check is irrelevant (it's a news article, not a photo).
            const isTransferNews = /signs?|joins?|transfers?|announces?|agreement|deal|completed|confirmed|here we go|move to|has joined|has signed/i.test(combinedText);
            // Clear any main-path jersey mismatch if it's a transfer announcement, screenshot, or trusted source
            if (jerseyMismatch && (isScreenshot || foundSource || isTransferNews)) {
                jerseyMismatch = null;
            }

            // Inject Gemini result if available
            if (aiOverride) {
                const aiFlag = `🤖 Gemini AI Detector: ${aiOverride.ai_probability}% probability of AI generation`;
                fallback.red_flags.push(aiFlag);
                if (aiOverride.ai_artifacts?.length) {
                    aiOverride.ai_artifacts.forEach(a => fallback.red_flags.push(`⚠️ AI artifact: ${a}`));
                }
                if (!aiOverride.softWarning) {
                    fallback.credibility_score = Math.min(fallback.credibility_score, aiOverride.credibility_score);
                    fallback.verdict = aiOverride.verdict;
                }
                if (aiOverride.ai_reasoning) {
                    fallback.summary = `${aiOverride.ai_reasoning} ${fallback.summary}`;
                }
            }

            // Gemini positive boost: when Groq is down but Gemini independently
            // says the image is Real/Likely Real with Medium+ confidence, raise score.
            // Soft warnings (aiOverride.softWarning) don't block this boost.
            // Screenshots bypass all heuristic guards.
            const hardOverride = aiOverride && !aiOverride.softWarning;
            const heuristicBlock = !isScreenshot && (portraitWarning || physiqueWarning || jerseyMismatch);
            // Selfie/portrait category: Gemini is notoriously bad at detecting its own
            // generated portraits — don't boost when Groq is down and the image is a
            // portrait, selfie, or headshot (Gemini's most common and least reliable category).
            const isSelfiePortrait = (aiDetection?.imageCategory || '').match(/mirror selfie|portrait|headshot/i);
            if (aiDetection && !hardOverride && !heuristicBlock && !filenameWarning && !isSelfiePortrait) {
                const gv = (aiDetection.verdict || '').toLowerCase();
                const gc = (aiDetection.confidence || '').toLowerCase();
                if ((gv.includes('real') || gv === 'likely real') && (gc === 'high' || gc === 'medium')) {
                    const boost = gv === 'real' ? 85 : 75;
                    fallback.credibility_score = Math.max(fallback.credibility_score, boost);
                    fallback.verdict = aiDetection.verdict;
                    fallback.green_flags.push(`🤖 Gemini AI forensics classified this image as: ${aiDetection.verdict} (${aiDetection.confidence} confidence)`);
                    fallback.summary = `${aiDetection.reasoning || ''} ${fallback.summary}`.trim();
                }
            }
            // Selfie/portrait flagged by Gemini with false "Real" verdict: inject suspicion
            if (isSelfiePortrait && aiDetection && !filenameWarning) {
                const gv = (aiDetection.verdict || '').toLowerCase();
                if (gv.includes('real') || gv === 'likely real') {
                    fallback.red_flags.push('⚠️ Gemini classified this portrait/selfie as "Real", but AI models commonly misclassify their own generated portraits. Treat this verdict with caution.');
                    fallback.summary = `[Caution: Gemini portrait self-assessment may be unreliable] ${fallback.summary}`;
                }
            }

            // Screenshot boost: even if Gemini's verdict isn't "Real", a screenshot
            // category means it's a capture of real content — boost to 80.
            if (isScreenshot && fallback.credibility_score < 80 && !filenameWarning && !jerseyMismatch) {
                fallback.credibility_score = Math.min(85, Math.max(fallback.credibility_score, 80));
                if (fallback.verdict.startsWith('Unable to Complete')) {
                    fallback.verdict = 'Likely Real — Screenshot of Social Media Post';
                }
                fallback.green_flags.push('📸 Gemini classified this as a screenshot — a capture of real content, not AI-generated');
            }

            // Inject Nonescape model result (advisory only — never caps the score;
            // the model is unreliable off-distribution, see docs/eval-image.md)
            if (nonescapeInfo.redFlag) {
                fallback.red_flags.push(nonescapeInfo.redFlag);
            }

            // Face identification
            if (faceIdOverride) {
                fallback.green_flags.push(`👤 Face ID matched: ${faceIdOverride}`);
                if (!aiOverride) fallback.credibility_score = Math.max(45, fallback.credibility_score);
            }

            // Jersey mismatch — re-check using available text (user context, face ID,
            // Gemini reasoning) since Groq vision may have failed to describe the image.
            // Skip for: screenshots, trusted sources, or transfer-announcement text
            // (the jersey check is designed for PHOTOS of players wearing the wrong jersey,
            // not for text articles about transfers).
            if (!isScreenshot && !foundSource && !isTransferNews && !jerseyMismatch) {
                const fallbackText = [userContext, faceIdOverride, aiDetection?.description, aiDetection?.reasoning, aiDetection?.imageCategory].filter(Boolean).join(' ');
                if (fallbackText) jerseyMismatch = detectJerseyMismatch(fallbackText);
            }
            if (jerseyMismatch && !isScreenshot && !foundSource && !isTransferNews) {
                fallback.credibility_score = 8;
                fallback.verdict = `Fake / Manipulated — ${jerseyMismatch.player} never played for ${jerseyMismatch.jerseyTeam}`;
                fallback.red_flags.push(`⚽ Jersey mismatch: ${jerseyMismatch.player} has NEVER played for ${jerseyMismatch.jerseyTeam}`);
                fallback.red_flags.push(`📋 Known clubs: ${jerseyMismatch.knownClubs}`);
                fallback.summary = jerseyMismatch.mismatchMsg;
            }

            // Unverifiable identity: jersey visible but no player name matched
            if (unverifiableIdentity && !isScreenshot && !foundSource && fallback.credibility_score > 20) {
                fallback.credibility_score = Math.min(fallback.credibility_score, 15);
                fallback.verdict = 'Suspicious — Unverifiable Person in Team Jersey';
                fallback.red_flags.push(`⚽ The jersey shows ${unverifiableIdentity}, but who the person is could not be determined`);
            }

            fallback.credibility_score = Math.max(0, Math.min(100, fallback.credibility_score));
            return fallback;
        }
        } catch (fbErr) {
            logger.error('Fallback path threw:', fbErr);
            return {
                credibility_score: 40,
                verdict: 'Analysis Incomplete (Internal Error)',
                bias: 'Unknown',
                sentiment: 'Neutral',
                red_flags: ['An internal error occurred during fallback analysis.'],
                green_flags: [],
                summary: 'Groq vision unavailable and fallback encountered an error.',
                recommendations: ['Try again later.', 'Consider using Text or URL analysis as an alternative.'],
                is_trusted_source: false,
                trusted_source_name: '',
                face_identified: faceIdOverride || null,
            };
        }

        throw new Error(`Image analysis failed: ${detail}`, { cause: error });
    }
}

// ── Trusted source detection in fallback text (when Groq is down) ──
const KNOWN_TRUSTED_SOURCES = [
    'Fabrizio Romano', 'David Ornstein', 'Florian Plettenberg', 'Shams Charania', 'Adrian Wojnarowski',
    'Sky Sports', 'ESPN', 'BBC', 'Reuters', 'AP News', 'AFP', 'The Guardian', 'NYT', 'Washington Post',
    'Bloomberg', 'WSJ', 'Marques Brownlee', 'MKBHD', 'IGN', 'The Verge', 'TechCrunch', 'Jason Schreier',
    'WHO', 'CDC', 'NASA', 'Nature', 'Science Magazine', 'Neil deGrasse Tyson', 'Andrew Huberman',
    'Financial Times', 'The Economist',
];
const KNOWN_TRUSTED_PATTERNS = KNOWN_TRUSTED_SOURCES.map(s => new RegExp('\\b' + s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'i'));
export function extractTrustedSource(text) {
    if (!text) return null;
    for (let i = 0; i < KNOWN_TRUSTED_PATTERNS.length; i++) {
        if (KNOWN_TRUSTED_PATTERNS[i].test(text)) return KNOWN_TRUSTED_SOURCES[i];
    }
    return null;
}

// ── Sports context suspicion (when Face API is unavailable) ──
// If we couldn't run the Face API (faceIdOverride is null) AND the image
// description suggests sports content but no player+team combo was matched,
// return a suspicion flag. Pure function, easily testable.
// Sports keywords — keep only unambiguous sports terms.
// "field" is excluded (agricultural/nature context), "kit" excluded (tool/animal context),
// "club" excluded (social/dining context).
const SPORTS_KEYWORDS = /\b(jersey|stadium|football|soccer|player|team|match|pitch|goal|goalie|referee|manager|transfer|badge|sponsor|training|warm.?up|substitute|captain|captaincy)\b/gi;

// Anti-keywords: if any appear in the description, do NOT flag as sports
// Includes nature, agriculture, and selfie/portrait terms
const SPORTS_ANTI_KEYWORDS = /\b(cow|cattle|farm|farmer|agriculture|pasture|barn|hay|garden|landscape|scenery|village|rural|nature|hike|picnic|animal|dog|cow|crop|field|grass|meadow|orchard|park|selfie|mirror|portrait|headshot|bedroom|bathroom|sink|doorway|hallway)\b/i;

export function detectSportsSuspicion(faceIdOverride, jerseyMismatch, imageDescription, findTeamFn = findTeamInText, geminiCategory) {
    if (faceIdOverride || jerseyMismatch) return null;
    if (!imageDescription) return null;
    // If Gemini explicitly says this is a selfie or portrait, skip
    if (geminiCategory && /mirror selfie|portrait|headshot/i.test(geminiCategory)) return null;
    const lowerDesc = imageDescription.toLowerCase();
    // If the description has non-sports keywords, don't flag as sports
    if (SPORTS_ANTI_KEYWORDS.test(lowerDesc)) return null;
    // Use word-boundary regex to prevent substring false matches
    // Require 3+ keyword matches to reduce false positives on selfies
    // where Groq may hallucinate terms like "player" or "training"
    const matchCount = (lowerDesc.match(SPORTS_KEYWORDS) || []).length;
    if (matchCount < 3) return null;
    const teamName = findTeamFn(imageDescription);
    if (teamName) return null;
    return 'Sports image detected but the player or team could not be identified. Unverifiable sports images may be analyzed.';
}

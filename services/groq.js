import Groq from 'groq-sdk';
import dotenv from 'dotenv';
import { detectAiImage, computeAiOverride } from './aiDetector.js';
dotenv.config();

const groq = new Groq({ apiKey: process.env.GROQ_API_KEY });

/**
 * Rule-based heuristic: detects if a Step 1 image description mentions
 * an extreme/competition-level physique in a casual everyday setting.
 * This combo is a primary AI muscle-enhancement indicator.
 * Returns a warning string if triggered, null otherwise.
 */
function detectExtremePhysiqueCasualSetting(description) {
    if (!description) return null;
    const d = description.toLowerCase();

    // Words that signal an impressive, well-developed, or extreme physique
    const PHYSIQUE_WORDS = [
        'muscular', 'muscle', 'muscles', 'defined', 'definition',
        'shredded', 'ripped', 'jacked', 'vascular', 'veins',
        'six.?pack', 'six pack', 'abs', 'abdominals',
        'chest', 'pectoral', 'bicep', 'tricep', 'shoulder',
        'physique', 'athletic', 'fit', 'lean', 'toned',
        'well.built', 'built', 'buff', 'huge', 'massive',
        'impressive body', 'impressive physique', 'strong',
        'bodybuilder', 'fitness',
    ];

    // Words that signal a casual everyday (non-competition) setting
    const CASUAL_SETTING_WORDS = [
        'mirror', 'selfie', 'bathroom', 'bedroom', 'room', 'wall',
        'door', 'home', 'house', 'apartment', 'phone', 'holding',
        'casual', 'everyday', 'indoor', 'hallway', 'hotel',
    ];

    // Words that signal extreme/competition-level conditioning
    // (these alone should trigger the warning even without the casual setting check)
    const EXTREME_PHYSIQUE_WORDS = [
        'shredded', 'ripped', 'vascular', 'veins', 'six.?pack',
        'six pack', 'shredded physique', 'extreme', 'extraordinary',
        'exceptional', 'impressive', 'incredible', 'amazing', 'remarkable',
        'competition', 'stage', 'professional bodybuil',
    ];

    const hasPhysique = PHYSIQUE_WORDS.some(w => new RegExp(w).test(d));
    const hasCasualSetting = CASUAL_SETTING_WORDS.some(w => new RegExp(w).test(d));
    const hasExtremePhysique = EXTREME_PHYSIQUE_WORDS.some(w => new RegExp(w).test(d));

    if (hasExtremePhysique && hasCasualSetting) {
        return `Image description indicates an EXTREME/COMPETITION-LEVEL physique (e.g. shredded, vascular, six-pack) in a CASUAL everyday setting (mirror, bathroom, bedroom). This specific combination is statistically very unlikely to be a natural unedited photo. AI muscle enhancement apps commonly produce exactly this result.`;
    }

    if (hasPhysique && hasCasualSetting) {
        return `Image description indicates a HIGHLY MUSCULAR physique in a CASUAL mirror/home setting. This is a known pattern for AI-enhanced body photos. Apply extra scrutiny.`;
    }

    return null;
}

/**
 * Rule-based heuristic: detects AI-generated portrait/face photos.
 * AI image generators (Gemini, DALL-E, Midjourney) produce portraits with
 * very consistent signatures: uniform background, perfect centered framing,
 * studio lighting, and no real-world environmental context.
 * Returns a warning string if triggered, null otherwise.
 */
function detectAiPortrait(description) {
    if (!description) return null;
    const d = description.toLowerCase();

    // ── Background detection (flexible: color word OR descriptor word near "background") ──
    // Check for neutral/uniform background keywords individually
    const BG_COLOR_WORDS = ['gray', 'grey', 'white', 'beige', 'neutral', 'plain', 'solid', 'uniform', 'seamless', 'simple', 'clean', 'blank', 'muted'];
    const BG_CONTEXT_WORDS = ['background', 'backdrop', 'wall', 'surface'];
    const hasBgColor = BG_COLOR_WORDS.some(w => d.includes(w));
    const hasBgContext = BG_CONTEXT_WORDS.some(w => d.includes(w));
    const hasUniformBg = hasBgColor && hasBgContext;

    // ── Portrait/headshot framing detection ──
    const PORTRAIT_SIGNALS = [
        'portrait', 'headshot', 'head shot', 'close-up', 'close up',
        'facing', 'looking at', 'looking directly', 'direct gaze',
        'front-facing', 'face forward', 'centered', 'centered in',
        'passport', 'id photo', 'profile photo', 'professional photo',
        'studio photo', 'head and shoulders',
    ];
    const hasPortraitFraming = PORTRAIT_SIGNALS.some(w => d.includes(w));

    // ── "Too perfect" quality signals ──
    const PERFECT_SIGNALS = [
        'smooth skin', 'clear skin', 'perfect skin', 'flawless', 'blemish',
        'studio lighting', 'professional lighting', 'even lighting', 'well-lit', 'well lit',
        'sharp', 'high quality', 'photorealistic', 'realistic', 'detailed',
        'no visible imperfection', 'no obvious',
    ];
    const hasPerfectQuality = PERFECT_SIGNALS.some(w => d.includes(w));

    // ── Real-world context (reduces AI suspicion) ──
    const REAL_CONTEXT = [
        'mirror', 'selfie', 'phone', 'holding', 'friends', 'family',
        'outdoors', 'outside', 'street', 'crowd', 'event', 'concert',
        'sports', 'jersey', 'gym', 'office', 'classroom', 'restaurant',
        'car', 'park', 'beach', 'nature',
    ];
    const hasRealContext = REAL_CONTEXT.some(w => d.includes(w));

    // ── Face/person signals ──
    const FACE_SIGNALS = ['man', 'woman', 'person', 'individual', 'face', 'young', 'male', 'female', 'subject'];
    const hasFace = FACE_SIGNALS.some(w => d.includes(w));

    // STRONG trigger: uniform bg + portrait framing + face + no real context
    if (hasUniformBg && hasPortraitFraming && hasFace && !hasRealContext) {
        return `Image matches the classic AI-generated portrait signature: uniform/neutral background, centered forward-facing portrait, no real-world environmental context. AI image generators (Gemini, DALL-E, Midjourney) consistently produce this exact pattern.`;
    }

    // MODERATE trigger: uniform bg + face + perfect quality + no real context
    if (hasUniformBg && hasFace && hasPerfectQuality && !hasRealContext) {
        return `Image has a uniform neutral background with studio-quality rendering, no real-world context — a common signature of AI portrait generators.`;
    }

    // WEAK trigger: only uniform bg + face (could be a real ID/passport photo — less confident)
    if (hasUniformBg && hasFace && !hasRealContext && !hasPortraitFraming) {
        // Don't cap hard, just note it
        return null;
    }

    return null;
}

/**
 * Rule-based heuristic: detects if the original filename contains obvious AI generator artifacts.
 */
function detectAiFilename(filename) {
    if (!filename) return null;
    const f = filename.toLowerCase();
    
    const AI_KEYWORDS = [
        'midjourney', 'dall-e', 'dalle', 'stable-diffusion', 'stablediffusion',
        'flux', 'comfyui', 'chatgpt', 'ai-generated', 'ai_generated',
        'generated_by', 'deepfake', 'faceapp', 'bing image creator', 'nightcafe'
    ];
    
    for (const keyword of AI_KEYWORDS) {
        if (f.includes(keyword)) {
            return `The original filename ("${filename}") explicitly contains the name of an AI image generator or AI tool ("${keyword}"). This is extremely strong evidence that the image is AI-generated.`;
        }
    }
    
    return null;
}

async function searchWeb(query) {
    try {
        const response = await fetch('https://html.duckduckgo.com/html/', {
            method: 'POST',
            headers: { 
                'Content-Type': 'application/x-www-form-urlencoded',
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            },
            body: 'q=' + encodeURIComponent(query),
            signal: AbortSignal.timeout(10000) // 10s timeout
        });
        const html = await response.text();
        const snippetRegex = /<a class="result__snippet[^>]*>(.*?)<\/a>/gi;
        let match;
        const snippets = [];
        while ((match = snippetRegex.exec(html)) !== null && snippets.length < 3) {
            let text = match[1].replace(/<[^>]+>/g, '').replace(/&quot;/g, '"').replace(/&#x27;/g, "'").replace(/&amp;/g, '&').trim();
            if (text) snippets.push(text);
        }
        return snippets.length ? snippets.join('\n- ') : "No recent news found.";
    } catch (err) {
        console.error("Search failed:", err.message);
        return "No recent news found.";
    }
}

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
`;

export async function analyzeContent(text) {
    // Truncate query for DuckDuckGo to avoid massive payload errors (DuckDuckGo expects short queries)
    const searchQuery = text.length > 300 ? text.substring(0, 300) : text;
    const searchResults = await searchWeb(searchQuery);

    // Truncate text for Llama model to prevent 413 Payload Too Large / Token Rate Limits
    // The TPM limit is 12000 tokens. Safely truncate to 15,000 characters.
    let contentToAnalyze = text;
    if (contentToAnalyze.length > 15000) {
        contentToAnalyze = contentToAnalyze.substring(0, 15000) + "\n...[Content truncated due to length]...";
    }

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

    Text to analyze:
    "${contentToAnalyze}"

    ${ANALYSIS_SCHEMA}
    `;

    try {
        const completion = await groq.chat.completions.create({
            messages: [{ role: "user", content: prompt }],
            model: "llama-3.3-70b-versatile",
            response_format: { type: "json_object" },
            temperature: 0.1,
        });

        const responseText = completion.choices[0]?.message?.content || '{}';
        let result = JSON.parse(responseText);
        
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
                if (localAiResult && localAiResult.fake_probability !== undefined) {
                    const fakeProb = Math.round(localAiResult.fake_probability * 100);
                    if (localAiResult.is_fake) {
                        result.red_flags = result.red_flags || [];
                        result.red_flags.push(`🤖 Local ML Model predicts Fake News with ${fakeProb}% probability.`);
                        if (result.credibility_score > 50) {
                            result.credibility_score = 50;
                            result.verdict = 'Mixed / Disputed (Local ML Model flagged as Fake)';
                        }
                    } else {
                        result.green_flags = result.green_flags || [];
                        result.green_flags.push(`✅ Local ML Model predicts Real News with ${Math.round(localAiResult.real_probability * 100)}% probability.`);
                    }
                }
            }
        } catch (e) {
            console.log("Local Python ML API not reachable or timed out, skipping ML score...");
        }

        return result;
    } catch (error) {
        console.error("Groq text analysis error:", error);
        throw error;
    }
}

// Vision models to try in order — if one is over capacity, fall back to the next
const VISION_MODELS = [
    'meta-llama/llama-4-scout-17b-16e-instruct',
    'meta-llama/llama-4-maverick-17b-128e-instruct',
    'llama-3.2-90b-vision-preview',
    'llama-3.2-11b-vision-preview',
];

async function groqVisionRequest(messages, maxTokens = 500) {
    let lastError;
    for (const model of VISION_MODELS) {
        try {
            console.log(`Trying vision model: ${model}`);
            const completion = await Promise.race([
                groq.chat.completions.create({
                    messages,
                    model,
                    temperature: 0.1,
                    max_tokens: maxTokens,
                }),
                new Promise((_, reject) => setTimeout(() => reject(new Error('Groq Timeout')), 45000))
            ]);
            console.log(`Vision model ${model} succeeded.`);
            return completion;
        } catch (err) {
            const msg = err?.error?.message || err?.message || '';
            const isOverCapacity = msg.includes('over capacity') || msg.includes('503') || (err?.status === 503);
            const isUnavailable = msg.includes('unavailable') || msg.includes('404') || (err?.status === 404);
            if (isOverCapacity || isUnavailable) {
                console.warn(`Model ${model} unavailable (${err?.status || 'err'}): ${msg.slice(0, 120)} — trying next model...`);
                lastError = err;
                continue;
            }
            // Non-capacity error — re-throw immediately
            throw err;
        }
    }
    throw lastError || new Error('All vision models are unavailable');
}

export async function analyzeImage(imageBase64, mimeType, userContext = '', originalName = '') {
    // Groq enforces a 4 MB limit on base64-encoded images
    const base64SizeBytes = Math.ceil(imageBase64.length * 3 / 4);
    const MAX_BASE64_BYTES = 4 * 1024 * 1024; // 4 MB
    if (base64SizeBytes > MAX_BASE64_BYTES) {
        throw new Error(`Image too large (${(base64SizeBytes / (1024 * 1024)).toFixed(1)} MB). Maximum is 4 MB. Please resize or compress the image.`);
    }

    const dataUrl = `data:${mimeType};base64,${imageBase64}`;

    // ── PRE-STEP: Dedicated AI image detection (Gemini Vision forensics) ──
    const aiDetection = await detectAiImage(imageBase64, mimeType);
    const aiOverride = computeAiOverride(aiDetection);

    // Build filename context note if the name looks suspicious
    const filenameNote = originalName
        ? `\n\nFILE METADATA: The original filename is "${originalName}". If this name contains references to AI tools (chatgpt, midjourney, dalle, stable diffusion, flux, etc.) or words like "fake", "edited", "enhanced", "generated" — treat it as a STRONG signal that the image is AI-generated or manipulated and score accordingly.`
        : '';

    // ── STEP 1: Rich visual description for context + search ──
    let imageDescription = '';
    console.log(`Starting Groq step 1 (Description)...`);
    const descStartTime = Date.now();
    try {
        const userContextNote = userContext
            ? `\n\nIMPORTANT — The user says this image: "${userContext}". Keep this in mind.`
            : '';

        const descCompletion = await groqVisionRequest([
            {
                role: "user",
                content: [
                    {
                        type: "text",
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
4. If the person is wearing a sports jersey: name who you think they are (only if you are highly confident — if uncertain, say "unidentified person"). Also state what team jersey they wear, and whether that person actually plays for that team.
5. If you are NOT confident about the person's identity, say so explicitly. DO NOT guess a famous person's name unless you are at least 80% certain.

--- SUMMARY ---
6. Write a 2-3 sentence summary of what the image shows and whether it appears authentic or manipulated.

Be very specific and honest about uncertainty.`
                    },
                    { type: "image_url", image_url: { url: dataUrl } }
                ]
            }
        ], 500);
        imageDescription = descCompletion.choices[0]?.message?.content || '';
        console.log(`Groq Description completed in ${Date.now() - descStartTime}ms. Output length: ${imageDescription.length}`);
    } catch (err) {
        console.error(`Groq Description failed after ${Date.now() - descStartTime}ms:`, err.message);
        // Continue without description — fall back to visual-only analysis
    }

    // ── STEP 2: Web search to fact-check the image content ──
    let searchResults = "No recent news found.";
    if (imageDescription) {
        searchResults = await searchWeb(imageDescription);
    }

    // ── STEP 2.5: Rule-based heuristic ──
    const physiqueWarning = detectExtremePhysiqueCasualSetting(imageDescription);
    if (physiqueWarning) {
        console.log(`⚠️ Physique heuristic triggered: ${physiqueWarning}`);
    }

    const portraitWarning = detectAiPortrait(imageDescription);
    if (portraitWarning) {
        console.log(`⚠️ Portrait heuristic triggered: ${portraitWarning}`);
    }

    const filenameWarning = detectAiFilename(originalName);
    if (filenameWarning) {
        console.log(`⚠️ Filename heuristic triggered: ${filenameWarning}`);
    }

    const anyHeuristicWarning = physiqueWarning || portraitWarning || filenameWarning;

    // ── STEP 3: Full analysis with both visual + factual context ──
    const userContextSection = userContext || filenameNote || anyHeuristicWarning
        ? `--- USER-PROVIDED CONTEXT (TREAT AS A STRONG SIGNAL) ---
    ${userContext ? `The person who uploaded this image says: "${userContext}"` : ''}
    ${filenameNote}
    ${physiqueWarning ? `\n⚠️ AUTOMATIC HEURISTIC WARNING (MUSCLE): ${physiqueWarning}\nThis combination (extreme competition-level physique + casual everyday setting) is a PRIMARY indicator of AI muscle enhancement. You MUST reflect this suspicion in your score and verdict. Score should be 30 or lower.` : ''}
    ${portraitWarning ? `\n⚠️ AUTOMATIC HEURISTIC WARNING (PORTRAIT): ${portraitWarning}\nThis image matches the signature of AI-generated portrait photos from tools like Gemini Image, DALL-E, and Midjourney. You MUST reflect this suspicion. Score should be 25 or lower unless you find specific real-world evidence this is genuine.` : ''}
    ${filenameWarning ? `\n⚠️ AUTOMATIC HEURISTIC WARNING (FILENAME): ${filenameWarning}\nYou MUST score this as extremely low credibility (1-10) because the filename itself reveals it is an AI generation.` : ''}
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
      If the person IS a highly-recognized athlete in a sports jersey:
      1. Confirm their identity (only if 80%+ certain)
      2. Confirm what team jersey they wear
      3. Cross-reference: does this player actually play for that team?
      4. If there's a mismatch → score 5-20, verdict "Fake / Manipulated - Player Not at This Club"

    --- VISUAL ANALYSIS FROM STEP 1 ---
    ${imageDescription || 'No description available.'}

    --- LIVE WEB SEARCH CONTEXT ---
    ${searchResults}
    --------------------------------------------------------

    ${ANALYSIS_SCHEMA}
    `;

    try {
        console.log(`Starting Groq step 3 (Final Analysis)...`);
        const analysisStartTime = Date.now();
        const completion = await groqVisionRequest([
            {
                role: "user",
                content: [
                    { type: "text", text: prompt },
                    { type: "image_url", image_url: { url: dataUrl } }
                ]
            }
        ], 1024);
        console.log(`Groq Analysis completed in ${Date.now() - analysisStartTime}ms.`);

        let responseText = completion.choices[0]?.message?.content || '{}';
        // Clean markdown blocks if vision model ignores json_object format
        responseText = responseText.replace(/```json/g, '').replace(/```/g, '').trim();
        const llmResult = JSON.parse(responseText);

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
                console.log(`Soft AI warning: original score ${originalScore} → adjusted ${adjustedScore} (Gemini: ${aiOverride.ai_probability}%)`);
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

        // ── HEURISTIC SCORE CAP ──
        // If either heuristic fired and Groq still returned a high score, cap it.
        // Groq often ignores prompt warnings and returns high scores anyway.

        if (portraitWarning && (llmResult.credibility_score ?? 100) > 30) {
            const cappedScore = 15;
            console.log(`⚠️ Portrait heuristic cap applied: score ${llmResult.credibility_score} → ${cappedScore}`);
            return {
                ...llmResult,
                credibility_score: cappedScore,
                verdict: 'Likely AI-Generated Portrait',
                red_flags: [
                    '🤖 AI portrait signature detected: uniform/neutral background + centered face + studio lighting + no real-world context',
                    'This exact composition is the default output of AI image generators (Gemini Image, DALL-E, Midjourney, Stable Diffusion)',
                    ...(llmResult.red_flags || []),
                ],
                green_flags: [],
                summary: `Automatic heuristic: ${portraitWarning} ${llmResult.summary || ''}`.trim(),
            };
        }

        if (physiqueWarning && (llmResult.credibility_score ?? 100) > 35) {
            const cappedScore = 20;
            console.log(`⚠️ Physique heuristic cap applied: score ${llmResult.credibility_score} → ${cappedScore}`);
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
            console.log(`⚠️ Filename heuristic cap applied: score ${llmResult.credibility_score} → ${cappedScore}`);
            return {
                ...llmResult,
                credibility_score: cappedScore,
                verdict: 'Confirmed AI-Generated',
                red_flags: [
                    `🤖 The file's original name explicitly reveals it was created by an AI tool`,
                    ...(llmResult.red_flags || []),
                ],
                green_flags: [],
                summary: `Automatic heuristic: ${filenameWarning} ${llmResult.summary || ''}`.trim(),
            };
        }

        // ── LOCAL IMAGE CLASSIFIER (animal + human) ──
        // Calls the ONNX model (cat | dog | humans | wild).
        // • Human detected  → inject a deepfake/AI scrutiny warning into red_flags
        //                     and note the classification in green_flags if score is already high
        // • Animal detected → inject a green flag with the classification result
        try {
            const FormData = (await import('form-data')).default;
            const form = new FormData();
            // Re-encode base64 to buffer to send as a file upload
            const imageBuffer = Buffer.from(imageBase64, 'base64');
            form.append('file', imageBuffer, { filename: 'image.jpg', contentType: mimeType });
            const animalRes = await fetch('http://127.0.0.1:8001/predict_animal', {
                method: 'POST',
                body: form,
                headers: form.getHeaders(),
                signal: AbortSignal.timeout(5000)
            });
            if (animalRes.ok) {
                const animalResult = await animalRes.json();
                if (animalResult?.predicted_class && animalResult?.confidence > 0.5) {
                    const pct = Math.round(animalResult.confidence * 100);

                    if (animalResult.is_human) {
                        // Human detected — flag for AI deepfake / body-enhancement scrutiny
                        llmResult.red_flags = llmResult.red_flags || [];
                        llmResult.red_flags.push(
                            `🧠 Human Detector (${pct}% confidence): This image contains a person. ` +
                            `AI deepfakes, face swaps, and body-enhancement edits are most common in ` +
                            `human photos — scrutinise skin texture, lighting consistency, and edge artifacts carefully.`
                        );

                        // If the overall score is suspiciously high for a human photo, apply a light penalty
                        if ((llmResult.credibility_score ?? 100) > 70 && !aiOverride) {
                            const humanPenalty = Math.round((pct / 100) * 10); // up to -10 pts
                            llmResult.credibility_score = Math.max(30, (llmResult.credibility_score ?? 80) - humanPenalty);
                            llmResult.summary = (llmResult.summary || '') +
                                ` [Human Classifier applied a ${humanPenalty}-pt AI-scrutiny adjustment.]`;
                        }
                    } else {
                        // Animal detected — positive signal (real animals are rarely deepfaked)
                        llmResult.green_flags = llmResult.green_flags || [];
                        llmResult.green_flags.push(
                            `🐾 Image Classifier: Detected a "${animalResult.predicted_class}" with ${pct}% confidence.`
                        );
                    }
                }
            }
        } catch (e) {
            console.log('Image classifier API not reachable, skipping...');
        }

        return llmResult;
    } catch (error) {
        const detail = error?.error?.message || error?.message || 'Unknown error';
        console.error("Groq image analysis error:", detail, error);
        throw new Error(`Image analysis failed: ${detail}`);
    }
}
